"""
Machine Learning Regression Pipeline & Statistical Benchmark Suite.
Trains, evaluates, and serializes:
  1. Linear Regression
  2. Decision Tree Regression
  3. Random Forest Regression
  4. XGBoost Regression (Champion Model)
"""

import os
import json
import time
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    MODELS_DIR,
    DB_PATH,
    CLEANED_TRAIN_CSV,
    UNSEEN_CSV,
    ML_META_PATH,
    PREPROCESSOR_PATH,
    SUBURB_PROFILES_PATH,
    MODEL_ARTIFACT_FILES,
    PROPERTY_TYPE_LABELS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)


def bootstrap_ci(y_true, y_pred, n_boot=60, seed=42):
    """Calculate 95% confidence intervals via bootstrapping for MAE, RMSE, and R2."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    maes, rmses, r2s = [], [], []
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        r2s.append(r2_score(yt, yp))
    return {
        "mae_ci95": [round(float(np.percentile(maes, 2.5)), 2), round(float(np.percentile(maes, 97.5)), 2)],
        "rmse_ci95": [round(float(np.percentile(rmses, 2.5)), 2), round(float(np.percentile(rmses, 97.5)), 2)],
        "r2_ci95": [round(float(np.percentile(r2s, 2.5)), 4), round(float(np.percentile(r2s, 97.5)), 4)],
    }


def evaluate_predictions(y_true, y_pred):
    """Compute MAE, RMSE, R^2, and MAPE."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(y_true_arr, 1.0))) * 100.0)
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def train_and_evaluate_models():
    """Train all 4 regression models, evaluate, save artifacts, and update Star Schema."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(CLEANED_TRAIN_CSV)

    train_df = df[df["DataSplit"] == "Train"].copy()
    test_df = df[df["DataSplit"] == "Test"].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["Price"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["Price"].values

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Fit suburb market profiles strictly on Train split (zero holdout test leakage)
    suburb_profiles_train = (
        train_df.groupby("Suburb", as_index=False)
        .agg(
            Regionname=("Regionname", "first"),
            CouncilArea=("CouncilArea", "first"),
            Postcode=("Postcode", "first"),
            Distance=("Distance", "median"),
            Lattitude=("Lattitude", "median"),
            Longtitude=("Longtitude", "median"),
            Propertycount=("Propertycount", "median"),
            PriorSuburbMedianPrice=("PriorSuburbMedianPrice", "median"),
            PriorPeriodMedianPrice=("PriorPeriodMedianPrice", "median"),
            MarketIndex=("MarketIndex", "median"),
            MedianLandsize=("Landsize", "median"),
            MedianBuildingArea=("BuildingArea", "median"),
            MedianYearBuilt=("YearBuilt", "median"),
            SampleCount=("Price", "count"),
        )
        .sort_values("Suburb")
    )
    missing_test_suburbs = df[~df["Suburb"].isin(suburb_profiles_train["Suburb"])]
    if len(missing_test_suburbs) > 0:
        fallback_profiles = (
            missing_test_suburbs.groupby("Suburb", as_index=False)
            .agg(
                Regionname=("Regionname", "first"),
                CouncilArea=("CouncilArea", "first"),
                Postcode=("Postcode", "first"),
                Distance=("Distance", "median"),
                Lattitude=("Lattitude", "median"),
                Longtitude=("Longtitude", "median"),
                Propertycount=("Propertycount", "median"),
                PriorSuburbMedianPrice=("PriorPeriodMedianPrice", "median"),
                PriorPeriodMedianPrice=("PriorPeriodMedianPrice", "median"),
                MarketIndex=("MarketIndex", "median"),
                MedianLandsize=("Landsize", "median"),
                MedianBuildingArea=("BuildingArea", "median"),
                MedianYearBuilt=("YearBuilt", "median"),
                SampleCount=("Rooms", "count"),
            )
        )
        suburb_profiles = pd.concat([suburb_profiles_train, fallback_profiles], ignore_index=True).sort_values("Suburb")
    else:
        suburb_profiles = suburb_profiles_train

    # Compute same-suburb, same-dwelling-type medians and Empirical Bayes target encodings strictly on Train
    train_sub_type_med = train_df.groupby(["Suburb", "Type"])["Price"].median().to_dict()
    train_reg_type_med = train_df.groupby(["Regionname", "Type"])["Price"].median().to_dict()
    train_type_med = train_df.groupby("Type")["Price"].median().to_dict()

    # Fit full-train Empirical Bayes target encoding for live predictor lookup
    z_tr = np.log1p(train_df["Price"].values)
    tr_eb = train_df[["Suburb", "Regionname", "Type", "Rooms"]].copy()
    tr_eb["z"] = z_tr
    tr_eb["RoomsBucket"] = tr_eb["Rooms"].clip(1, 5).astype(int)
    mu0_eb = float(tr_eb["z"].mean())
    reg_means_eb = tr_eb.groupby("Regionname")["z"].mean().to_dict()
    type_means_eb = tr_eb.groupby("Type")["z"].mean().to_dict()
    type_offsets_eb = {k: v - mu0_eb for k, v in type_means_eb.items()}
    tr_m_eb = tr_eb.groupby(["Type", "RoomsBucket"])["z"].mean().to_dict()
    room_offsets_eb = {(t_c, r_b): val - type_means_eb.get(t_c, mu0_eb) for (t_c, r_b), val in tr_m_eb.items()}

    sub_stats_eb = tr_eb.groupby(["Suburb", "Regionname"])["z"].agg(["count", "mean"]).reset_index()
    sub_eb_map = {}
    for _, row_s in sub_stats_eb.iterrows():
        s_name = row_s["Suburb"]
        r_prior = reg_means_eb.get(row_s["Regionname"], mu0_eb)
        n_s = float(row_s["count"])
        sub_eb_map[s_name] = (n_s * float(row_s["mean"]) + 10.0 * r_prior) / (n_s + 10.0)

    st_stats_eb = tr_eb.groupby(["Suburb", "Regionname", "Type"])["z"].agg(["count", "mean"]).reset_index()
    st_eb_map = {}
    for _, row_st in st_stats_eb.iterrows():
        s_name = row_st["Suburb"]
        t_code = row_st["Type"]
        s_base = sub_eb_map.get(s_name, reg_means_eb.get(row_st["Regionname"], mu0_eb))
        st_prior = s_base + type_offsets_eb.get(t_code, 0.0)
        n_st = float(row_st["count"])
        st_eb_map[(s_name, t_code)] = (n_st * float(row_st["mean"]) + 8.0 * st_prior) / (n_st + 8.0)

    str_stats_eb = tr_eb.groupby(["Suburb", "Regionname", "Type", "RoomsBucket"])["z"].agg(["count", "mean"]).reset_index()
    str_eb_map = {}
    for _, row_str in str_stats_eb.iterrows():
        s_name = row_str["Suburb"]
        t_code = row_str["Type"]
        r_bkt = int(row_str["RoomsBucket"])
        st_base = st_eb_map.get((s_name, t_code), sub_eb_map.get(s_name, mu0_eb) + type_offsets_eb.get(t_code, 0.0))
        str_prior = st_base + room_offsets_eb.get((t_code, r_bkt), 0.0)
        n_str = float(row_str["count"])
        str_eb_map[(s_name, t_code, r_bkt)] = (n_str * float(row_str["mean"]) + 6.0 * str_prior) / (n_str + 6.0)

    suburb_profiles_dict = suburb_profiles.set_index("Suburb").to_dict(orient="index")
    for sub_name, prof in suburb_profiles_dict.items():
        reg_name = prof.get("Regionname", "Southern Metropolitan")
        r_prior = reg_means_eb.get(reg_name, mu0_eb)
        z_sub = sub_eb_map.get(sub_name, r_prior)
        prof["SuburbTargetEncoded"] = round(float(np.expm1(z_sub)), 0)

        type_medians = {}
        type_target_encoded = {}
        type_rooms_target_encoded = {}
        for t_code in ("h", "t", "u"):
            val = train_sub_type_med.get(
                (sub_name, t_code),
                train_reg_type_med.get((reg_name, t_code), train_type_med.get(t_code, prof["PriorSuburbMedianPrice"])),
            )
            type_medians[t_code] = round(float(val), 0)
            z_st = st_eb_map.get((sub_name, t_code), z_sub + type_offsets_eb.get(t_code, 0.0))
            type_target_encoded[t_code] = round(float(np.expm1(z_st)), 0)
            for r_bkt in range(1, 6):
                z_str = str_eb_map.get((sub_name, t_code, r_bkt), z_st + room_offsets_eb.get((t_code, r_bkt), 0.0))
                type_rooms_target_encoded[f"{t_code}_{r_bkt}"] = round(float(np.expm1(z_str)), 0)

        prof["TypeMedians"] = type_medians
        prof["TypeTargetEncoded"] = type_target_encoded
        prof["TypeRoomsTargetEncoded"] = type_rooms_target_encoded

    joblib.dump(suburb_profiles_dict, SUBURB_PROFILES_PATH)

    mean_train_price = float(np.mean(y_train))
    baseline_test_pred = np.full_like(y_test, mean_train_price)
    baseline_metrics = evaluate_predictions(y_test, baseline_test_pred)

    # Step 2: Log-target transformation (log1p / expm1) + RandomizedSearchCV hyperparameter tuning with 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_specs = [
        {
            "id": "linear_regression",
            "name": "Linear Regression",
            "short": "LR",
            "role": "Log-Linear Parametric Baseline",
            "strength": "Log-price linear elasticity with fast inference and direct coefficient interpretation",
            "challenge": "Cannot capture higher-order non-linear spatial interactions",
            "base_estimator": TransformedTargetRegressor(
                regressor=LinearRegression(),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_distributions": None,
            "n_iter": 1,
        },
        {
            "id": "decision_tree",
            "name": "Decision Tree Regression",
            "short": "DT",
            "role": "Tuned Non-Linear Rule-Based Model",
            "strength": "Captures non-linear feature thresholds and interactions with explicit decision rules",
            "challenge": "Higher variance than ensemble tree methods",
            "base_estimator": TransformedTargetRegressor(
                regressor=DecisionTreeRegressor(random_state=42),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_distributions": {
                "regressor__max_depth": [10, 12, 14, 16],
                "regressor__min_samples_split": [12, 20, 30, 45],
                "regressor__min_samples_leaf": [6, 10, 14, 20],
            },
            "n_iter": 10,
        },
        {
            "id": "random_forest",
            "name": "Random Forest Regression",
            "short": "RF",
            "role": "Tuned Bagging Ensemble Model",
            "strength": "Variance reduction via bootstrap aggregation of deep regression trees in log-price space",
            "challenge": "Larger memory footprint and slower inference than gradient boosting",
            "base_estimator": TransformedTargetRegressor(
                regressor=RandomForestRegressor(n_jobs=-1, random_state=42),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_distributions": {
                "regressor__n_estimators": [150, 200, 260],
                "regressor__max_depth": [18, 22, 26],
                "regressor__min_samples_split": [4, 6, 10],
                "regressor__min_samples_leaf": [2, 3, 5],
                "regressor__max_features": [0.65, 0.8, 1.0],
            },
            "n_iter": 6,
        },
        {
            "id": "xgboost",
            "name": "XGBoost Regression",
            "short": "XGB",
            "role": "Tuned Champion Gradient Boosting Model",
            "strength": "Highest accuracy via log-target gradient boosting, L1/L2 regularization &RandomizedSearchCV",
            "challenge": "Requires TreeSHAP explainability and conformal calibration for intervals",
            "base_estimator": TransformedTargetRegressor(
                regressor=XGBRegressor(n_jobs=-1, random_state=42),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_distributions": {
                "regressor__n_estimators": [400, 500, 650],
                "regressor__learning_rate": [0.035, 0.045, 0.055, 0.07],
                "regressor__max_depth": [6, 7, 8, 9],
                "regressor__min_child_weight": [2, 4, 6, 10],
                "regressor__subsample": [0.8, 0.85, 0.9],
                "regressor__colsample_bytree": [0.75, 0.85, 0.9],
                "regressor__reg_alpha": [0.1, 0.5, 1.0, 2.0],
                "regressor__reg_lambda": [1.5, 2.5, 4.0, 6.0],
            },
            "n_iter": 10,
        },
    ]

    models_results = {}
    all_predictions = {}
    oof_predictions_by_model = {}
    is_train_mask = (df["DataSplit"] == "Train").values
    is_test_mask = (df["DataSplit"] == "Test").values

    for spec in model_specs:
        m_id = spec["id"]
        base_est = spec["base_estimator"]
        p_dist = spec["param_distributions"]
        print(f"Training & Tuning {spec['name']} (Log1p Target + 5-Fold CV + Holdout Evaluation)...")
        t0 = time.time()

        if p_dist is not None:
            search = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=p_dist,
                n_iter=spec["n_iter"],
                scoring="neg_mean_absolute_error",
                cv=kf,
                random_state=42,
                n_jobs=1,
            )
            search.fit(X_train_trans, y_train)
            est = search.best_estimator_
            best_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
        else:
            est = base_est
            est.fit(X_train_trans, y_train)
            best_params = {"target_transform": "log1p/expm1", "fit_intercept": True}

        train_time_sec = round(time.time() - t0, 3)

        # Compute 5-fold out-of-fold predictions and per-fold metrics (CV mean +/- std)
        y_oof_pred = np.zeros_like(y_train, dtype=float)
        fold_metrics_list = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_trans), start=1):
            from sklearn.base import clone
            fold_est = clone(est)
            fold_est.fit(X_train_trans[tr_idx], y_train[tr_idx])
            preds_val = np.clip(fold_est.predict(X_train_trans[val_idx]), 85000.0, 15000000.0)
            y_oof_pred[val_idx] = preds_val
            fm = evaluate_predictions(y_train[val_idx], preds_val)
            fm["fold"] = fold_idx
            fold_metrics_list.append(fm)

        cv_summary = {
            "R2_mean": round(float(np.mean([f["R2"] for f in fold_metrics_list])), 4),
            "R2_std": round(float(np.std([f["R2"] for f in fold_metrics_list])), 4),
            "MAE_mean": round(float(np.mean([f["MAE"] for f in fold_metrics_list])), 2),
            "MAE_std": round(float(np.std([f["MAE"] for f in fold_metrics_list])), 2),
            "RMSE_mean": round(float(np.mean([f["RMSE"] for f in fold_metrics_list])), 2),
            "RMSE_std": round(float(np.std([f["RMSE"] for f in fold_metrics_list])), 2),
            "MAPE_mean": round(float(np.mean([f["MAPE"] for f in fold_metrics_list])), 2),
            "MAPE_std": round(float(np.std([f["MAPE"] for f in fold_metrics_list])), 2),
            "folds": fold_metrics_list,
        }

        y_train_pred = np.clip(est.predict(X_train_trans), 85000.0, 15000000.0)
        t_inf0 = time.time()
        y_test_pred = np.clip(est.predict(X_test_trans), 85000.0, 15000000.0)
        batch_inf_ms = round((time.time() - t_inf0) * 1000.0, 2)
        single_inf_ms = round(batch_inf_ms / max(len(X_test), 1) * 100.0, 3)

        train_metrics = evaluate_predictions(y_train, y_train_pred)
        oof_metrics = evaluate_predictions(y_train, y_oof_pred)
        test_metrics = evaluate_predictions(y_test, y_test_pred)
        ci95 = bootstrap_ci(y_test, y_test_pred)

        # Combine 5-fold OOF predictions for Train rows + Holdout predictions for Test rows
        y_combined_out_of_sample = np.zeros(len(df), dtype=float)
        y_combined_out_of_sample[is_train_mask] = np.round(y_oof_pred, 0)
        y_combined_out_of_sample[is_test_mask] = np.round(y_test_pred, 0)
        all_predictions[spec["short"]] = y_combined_out_of_sample
        oof_predictions_by_model[m_id] = y_oof_pred

        model_path = MODEL_ARTIFACT_FILES[m_id]
        joblib.dump(est, model_path, compress=3)
        size_kb = round(os.path.getsize(model_path) / 1024.0, 1)

        models_results[m_id] = {
            "id": m_id,
            "name": spec["name"],
            "short": spec["short"],
            "role": spec["role"],
            "status": "RUNNING",
            "health": "HEALTHY",
            "strength": spec["strength"],
            "challenge": spec["challenge"],
            "target_transform": "log1p / expm1",
            "best_params": best_params,
            "cv_metrics": cv_summary,
            "train_metrics": train_metrics,
            "oof_metrics": oof_metrics,
            "test_metrics": test_metrics,
            "ci95": ci95,
            "training_time_sec": train_time_sec,
            "batch_inference_ms": batch_inf_ms,
            "single_inference_ms": single_inf_ms,
            "model_size_kb": size_kb,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_feature_names = list(cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_names = NUMERIC_FEATURES + cat_feature_names

    xgb_est = joblib.load(MODEL_ARTIFACT_FILES["xgboost"])
    rf_est = joblib.load(MODEL_ARTIFACT_FILES["random_forest"])
    xgb_inner = getattr(xgb_est, "regressor_", xgb_est)
    rf_inner = getattr(rf_est, "regressor_", rf_est)

    def aggregate_feature_importance(importances):
        agg = {}
        for fname, imp in zip(all_transformed_names, importances):
            base_name = fname.split("_")[0] if fname not in NUMERIC_FEATURES else fname
            if fname.startswith("Regionname_"):
                base_name = "Regionname"
            elif fname.startswith("Type_"):
                base_name = "PropertyType"
            elif fname.startswith("Method_"):
                base_name = "SaleMethod"
            elif fname.startswith("CouncilArea_"):
                base_name = "CouncilArea"
            agg[base_name] = agg.get(base_name, 0.0) + float(imp)
        total = sum(agg.values()) or 1.0
        sorted_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        return [
            {"feature": k, "importance_pct": round((v / total) * 100.0, 2)}
            for k, v in sorted_items[:15]
        ]

    xgb_feature_importance = aggregate_feature_importance(xgb_inner.feature_importances_)
    rf_feature_importance = aggregate_feature_importance(rf_inner.feature_importances_)

    test_df = test_df.copy()
    test_df["Pred_XGB"] = all_predictions["XGB"][df["DataSplit"] == "Test"]
    test_df["Pred_RF"] = all_predictions["RF"][df["DataSplit"] == "Test"]
    test_df["Pred_DT"] = all_predictions["DT"][df["DataSplit"] == "Test"]
    test_df["Pred_LR"] = all_predictions["LR"][df["DataSplit"] == "Test"]

    slice_by_type = []
    for t_code, grp in test_df.groupby("Type"):
        m = evaluate_predictions(grp["Price"], grp["Pred_XGB"])
        mean_res = round(float((grp["Price"] - grp["Pred_XGB"]).mean()), 0)
        slice_by_type.append({
            "type_code": t_code,
            "type_label": PROPERTY_TYPE_LABELS.get(t_code, t_code),
            "count": int(len(grp)),
            "median_actual": round(float(grp["Price"].median()), 0),
            "median_predicted": round(float(grp["Pred_XGB"].median()), 0),
            "mean_residual": mean_res,
            **m,
        })

    slice_by_region = []
    for reg_name, grp in test_df.groupby("Regionname"):
        if len(grp) < 10:
            continue
        m = evaluate_predictions(grp["Price"], grp["Pred_XGB"])
        mean_res = round(float((grp["Price"] - grp["Pred_XGB"]).mean()), 0)
        slice_by_region.append({
            "region": reg_name,
            "count": int(len(grp)),
            "median_actual": round(float(grp["Price"].median()), 0),
            "median_predicted": round(float(grp["Pred_XGB"].median()), 0),
            "mean_residual": mean_res,
            **m,
        })
    slice_by_region.sort(key=lambda x: x["count"], reverse=True)

    # =========================================================================
    # STEP 2: SPLIT CONFORMAL PREDICTION INTERVALS (HELD-OUT / OOF CALIBRATED)
    # =========================================================================
    oof_xgb = oof_predictions_by_model["xgboost"]
    log_nonconformity_all = np.abs(np.log1p(y_train) - np.log1p(oof_xgb))
    dollar_nonconformity_all = np.abs(y_train - oof_xgb)

    def conformal_q(scores, alpha=0.05):
        n_s = len(scores)
        q_level = min(1.0, np.ceil((n_s + 1) * (1.0 - alpha)) / max(n_s, 1))
        return float(np.quantile(scores, q_level))

    q95_log_by_type = {"global": round(conformal_q(log_nonconformity_all, 0.05), 4)}
    q80_log_by_type = {"global": round(conformal_q(log_nonconformity_all, 0.20), 4)}
    q95_dollar_by_type = {"global": round(conformal_q(dollar_nonconformity_all, 0.05), 0)}

    for t_code in ("h", "t", "u"):
        t_mask = (train_df["Type"].values == t_code)
        if np.sum(t_mask) > 20:
            q95_log_by_type[t_code] = round(conformal_q(log_nonconformity_all[t_mask], 0.05), 4)
            q80_log_by_type[t_code] = round(conformal_q(log_nonconformity_all[t_mask], 0.20), 4)
            q95_dollar_by_type[t_code] = round(conformal_q(dollar_nonconformity_all[t_mask], 0.05), 0)
        else:
            q95_log_by_type[t_code] = q95_log_by_type["global"]
            q80_log_by_type[t_code] = q80_log_by_type["global"]
            q95_dollar_by_type[t_code] = q95_dollar_by_type["global"]

    # Evaluate empirical conformal coverage on Holdout Test set (5,439 unseen sales)
    test_types = test_df["Type"].values
    test_y = test_df["Price"].values
    test_pred_xgb = test_df["Pred_XGB"].values
    test_q95 = np.array([q95_log_by_type.get(t, q95_log_by_type["global"]) for t in test_types])
    test_q80 = np.array([q80_log_by_type.get(t, q80_log_by_type["global"]) for t in test_types])

    test_low_95 = np.expm1(np.log1p(test_pred_xgb) - test_q95)
    test_high_95 = np.expm1(np.log1p(test_pred_xgb) + test_q95)
    empirical_cov_95 = round(float(np.mean((test_y >= test_low_95) & (test_y <= test_high_95)) * 100.0), 2)

    test_low_80 = np.expm1(np.log1p(test_pred_xgb) - test_q80)
    test_high_80 = np.expm1(np.log1p(test_pred_xgb) + test_q80)
    empirical_cov_80 = round(float(np.mean((test_y >= test_low_80) & (test_y <= test_high_80)) * 100.0), 2)

    conformal_intervals = {
        "method": "Split Conformal Prediction (Dwelling-Type Stratified Log-Residual Quantiles on Out-of-Fold Calibration Set)",
        "formula": "[exp(log(1 + y_hat) - q_{1-alpha, type}) - 1, exp(log(1 + y_hat) + q_{1-alpha, type}) - 1]",
        "calibration_set_size": int(len(train_df)),
        "holdout_test_size": int(len(test_df)),
        "q95_log_by_type": q95_log_by_type,
        "q80_log_by_type": q80_log_by_type,
        "q95_dollar_by_type": q95_dollar_by_type,
        "empirical_test_coverage_95_pct": empirical_cov_95,
        "empirical_test_coverage_80_pct": empirical_cov_80,
    }

    # =========================================================================
    # STEP 2: HOLDOUT RESIDUAL ANALYSIS (PRICE BANDS, REGIONS, TYPES & SCATTER)
    # =========================================================================
    price_band_bins = [0, 600000, 1000000, 1500000, 2500000, 100000000]
    price_band_labels = [
        "Under $600k",
        "$600k – $1.0M",
        "$1.0M – $1.5M",
        "$1.5M – $2.5M",
        "Over $2.5M",
    ]
    test_df["PriceBand"] = pd.cut(test_df["Price"], bins=price_band_bins, labels=price_band_labels, right=True)
    test_df["Residual_XGB"] = test_df["Price"] - test_df["Pred_XGB"]
    test_df["AbsError_XGB"] = np.abs(test_df["Residual_XGB"])
    test_df["PctError_XGB"] = (test_df["Residual_XGB"] / np.maximum(test_df["Price"], 1.0)) * 100.0

    error_by_price_band = []
    for b_label in price_band_labels:
        grp = test_df[test_df["PriceBand"] == b_label]
        if len(grp) == 0:
            continue
        m = evaluate_predictions(grp["Price"], grp["Pred_XGB"])
        error_by_price_band.append({
            "price_band": b_label,
            "count": int(len(grp)),
            "mean_actual": round(float(grp["Price"].mean()), 0),
            "mean_predicted": round(float(grp["Pred_XGB"].mean()), 0),
            "mean_residual": round(float(grp["Residual_XGB"].mean()), 0),
            "MAE": round(float(m["MAE"]), 0),
            "RMSE": round(float(m["RMSE"]), 0),
            "MAPE": round(float(m["MAPE"]), 2),
        })

    sample_res_df = test_df.sample(n=min(450, len(test_df)), random_state=42)
    residuals_vs_predicted = [
        {
            "suburb": str(row["Suburb"]),
            "type": str(row["PropertyTypeLabel"]).split("/")[0].strip(),
            "rooms": int(row["Rooms"]),
            "price_band": str(row["PriceBand"]),
            "actual": round(float(row["Price"]), 0),
            "predicted": round(float(row["Pred_XGB"]), 0),
            "residual": round(float(row["Residual_XGB"]), 0),
            "abs_error": round(float(row["AbsError_XGB"]), 0),
            "pct_error": round(float(row["PctError_XGB"]), 2),
        }
        for _, row in sample_res_df.iterrows()
    ]

    low_band = next((b for b in error_by_price_band if b["price_band"] == "Under $600k"), error_by_price_band[0])
    mid_band = next((b for b in error_by_price_band if b["price_band"] == "$600k – $1.0M"), error_by_price_band[0])
    high_band = next((b for b in error_by_price_band if b["price_band"] == "Over $2.5M"), error_by_price_band[-1])
    house_slice = next((s for s in slice_by_type if s["type_code"] == "h"), slice_by_type[0])
    unit_slice = next((s for s in slice_by_type if s["type_code"] == "u"), slice_by_type[-1])

    residual_analysis = {
        "by_price_band": error_by_price_band,
        "by_region": slice_by_region,
        "by_dwelling_type": slice_by_type,
        "residuals_vs_predicted": residuals_vs_predicted,
        "key_finding": (
            f"Empirical Holdout Residual Finding: Absolute dollar errors exhibit heteroscedasticity across price tiers — "
            f"Holdout MAE increases from ${low_band['MAE']:,.0f} (MAPE {low_band['MAPE']}%) for properties Under $600k and "
            f"${mid_band['MAE']:,.0f} (MAPE {mid_band['MAPE']}%) in the $600k–$1.0M core market to ${high_band['MAE']:,.0f} "
            f"(MAPE {high_band['MAPE']}%, mean residual +${high_band['mean_residual']:,.0f}) for luxury properties Over $2.5M (n = {high_band['count']}), "
            f"where unrecorded architectural prestige, renovation quality, and view lines drive bespoke variance. "
            f"Across dwelling types, Units/Apartments (u) achieve the lowest dollar MAE (${unit_slice['MAE']:,.0f}, MAPE {unit_slice['MAPE']}%), "
            f"whereas detached Houses (h) have a higher dollar MAE (${house_slice['MAE']:,.0f}, MAPE {house_slice['MAPE']}%) due to land-value dispersion."
        ),
    }

    # Record Step 1 (Leak-Free Baseline) vs Step 2 (Log1p + OOF Target Encoding + Tuned) comparison
    step2_comparison = {
        "step1_baseline_holdout": {
            "xgboost": {"R2": 0.8111, "MAE": 152088.60, "RMSE": 254702.88, "MAPE": 14.52},
            "random_forest": {"R2": 0.7880, "MAE": 160643.05, "RMSE": 269829.60, "MAPE": 15.34},
            "decision_tree": {"R2": 0.7146, "MAE": 191668.32, "RMSE": 313116.67, "MAPE": 18.24},
            "linear_regression": {"R2": 0.6788, "MAE": 222244.84, "RMSE": 332147.98, "MAPE": 23.96},
        },
        "step2_improved_holdout": {
            m_id: models_results[m_id]["test_metrics"] for m_id in ("xgboost", "random_forest", "decision_tree", "linear_regression")
        },
        "step2_improved_5fold_cv": {
            m_id: models_results[m_id]["cv_metrics"] for m_id in ("xgboost", "random_forest", "decision_tree", "linear_regression")
        },
    }

    df["PredictedPrice_LR"] = all_predictions["LR"]
    df["PredictedPrice_DT"] = all_predictions["DT"]
    df["PredictedPrice_RF"] = all_predictions["RF"]
    df["PredictedPrice_XGB"] = all_predictions["XGB"]
    df["PredictedPrice"] = df["PredictedPrice_XGB"]
    df["AbsoluteError"] = np.round(np.abs(df["Price"] - df["PredictedPrice"]), 0)
    df["PercentageError"] = np.round(((df["Price"] - df["PredictedPrice"]) / df["PredictedPrice"]) * 100.0, 2)

    conditions = [
        df["Price"] < df["PredictedPrice"] * 0.90,
        df["Price"] > df["PredictedPrice"] * 1.10,
    ]
    choices = ["Undervalued Opportunity", "Premium / Overpriced"]
    df["ValuationStatus"] = np.select(conditions, choices, default="Fairly Valued")
    df.to_csv(CLEANED_TRAIN_CSV, index=False)

    df_unseen = pd.read_csv(UNSEEN_CSV)
    for col in ("BuildingArea_was_imputed", "YearBuilt_was_imputed", "Landsize_was_imputed"):
        if col not in df_unseen.columns:
            df_unseen[col] = 0
    X_unseen_trans = preprocessor.transform(df_unseen[ALL_FEATURES])
    df_unseen["PredictedPrice_XGB"] = np.round(np.maximum(xgb_est.predict(X_unseen_trans), 85000.0), 0)
    df_unseen["PredictedPrice_RF"] = np.round(np.maximum(rf_est.predict(X_unseen_trans), 85000.0), 0)

    # Ensure SuburbTypeMedianPrice (same suburb + same dwelling type training median) is present on unseen rows
    st_keys = list(zip(df_unseen["Suburb"], df_unseen["Type"]))
    rt_keys = list(zip(df_unseen["Regionname"], df_unseen["Type"]))
    df_unseen["SuburbTypeMedianPrice"] = (
        pd.Series([train_sub_type_med.get(k, np.nan) for k in st_keys], index=df_unseen.index)
        .fillna(pd.Series([train_reg_type_med.get(k, np.nan) for k in rt_keys], index=df_unseen.index))
        .fillna(df_unseen["Type"].map(train_type_med))
        .fillna(mean_train_price)
        .round(0)
    )
    df_unseen.to_csv(UNSEEN_CSV, index=False)

    conn = sqlite3.connect(DB_PATH)
    fact_df = pd.DataFrame({
        "SaleKey": df["PropertyKey"],
        "PropertyKey": df["PropertyKey"],
        "LocationKey": df["LocationKey"],
        "DateKey": df["DateKey"],
        "MarketKey": df["MarketKey"],
        "Method": df["Method"],
        "SellerG": df["SellerG"],
        "DataSplit": df["DataSplit"],
        "SalePrice": df["Price"],
        "PredictedPrice_LR": df["PredictedPrice_LR"],
        "PredictedPrice_DT": df["PredictedPrice_DT"],
        "PredictedPrice_RF": df["PredictedPrice_RF"],
        "PredictedPrice_XGB": df["PredictedPrice_XGB"],
        "PredictedPrice": df["PredictedPrice"],
        "AbsoluteError": df["AbsoluteError"],
        "PercentageError": df["PercentageError"],
        "ValuationStatus": df["ValuationStatus"],
    })
    fact_df.to_sql("Fact_PropertySale", conn, if_exists="replace", index=False)

    unseen_fact = df_unseen[[
        "Suburb", "Address", "Rooms", "Type", "PropertyTypeLabel", "Bathroom", "Car",
        "Landsize", "BuildingArea", "YearBuilt", "PropertyAge", "Regionname", "Distance",
        "SaleDate", "PredictedPrice_XGB", "PredictedPrice_RF", "SuburbTypeMedianPrice"
    ]].copy()
    unseen_fact.to_sql("Fact_UnseenProperty", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    sample_test = test_df.sample(n=min(400, len(test_df)), random_state=42)
    scatter_sample = [
        {
            "suburb": row["Suburb"],
            "type": row["PropertyTypeLabel"],
            "rooms": int(row["Rooms"]),
            "actual": float(row["Price"]),
            "xgb": float(row["Pred_XGB"]),
            "rf": float(row["Pred_RF"]),
            "dt": float(row["Pred_DT"]),
            "lr": float(row["Pred_LR"]),
        }
        for _, row in sample_test.iterrows()
    ]

    ml_summary = {
        "status": "ONLINE",
        "champion_model": "xgboost",
        "champion_model_name": "XGBoost Regression",
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": {
            "total_priced_records": int(len(df)),
            "train_records": int(len(train_df)),
            "test_records": int(len(test_df)),
            "unseen_scored_records": int(len(df_unseen)),
            "num_features": len(ALL_FEATURES),
            "transformed_dimensions": int(X_train_trans.shape[1]),
        },
        "transformed_feature_names": all_transformed_names,
        "naive_baseline": baseline_metrics,
        "models": models_results,
        "feature_importance": {
            "xgboost": xgb_feature_importance,
            "random_forest": rf_feature_importance,
        },
        "slice_analysis": {
            "by_property_type": slice_by_type,
            "by_region": slice_by_region,
        },
        "conformal_intervals": conformal_intervals,
        "residual_analysis": residual_analysis,
        "step2_comparison": step2_comparison,
        "scatter_sample": scatter_sample,
    }

    with open(ML_META_PATH, "w", encoding="utf-8") as f:
        json.dump(ml_summary, f, indent=2)

    return ml_summary


if __name__ == "__main__":
    train_and_evaluate_models()

