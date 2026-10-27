"""
Automated Integration Test Suite for the Melbourne Property Valuation & ABI Platform.
Run with:
    python tests/test_platform_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web_app import app


def run_all_tests():
    client = app.test_client()

    r_home = client.get("/")
    assert r_home.status_code == 200, f"GET / failed: {r_home.status_code}"
    html_text = r_home.get_data(as_text=True)
    assert "Residential Property Price Prediction &amp; BI Decision Support" in html_text
    assert "Property Valuation (Live Predictor)" in html_text
    assert "BI Decision Support Dashboard" in html_text
    assert "Models &amp; Star Schema Pipeline" in html_text
    assert "Property Concierge" not in html_text
    assert "Fact_UnseenProperty" in html_text
    print(f"[PASS] GET / -> 200 OK ({len(r_home.data):,} bytes, UI headers & tabs verified)")

    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    st = r_status.get_json()
    assert st["active_models_count"] == 4
    print(f"[PASS] GET /api/status -> {st['overall_status']} ({st['active_models_count']}/{st['total_models_count']} ML models RUNNING)")

    r_opt = client.get("/api/options")
    assert r_opt.status_code == 200
    opt = r_opt.get_json()
    print(f"[PASS] GET /api/options -> {len(opt['suburbs'])} suburbs, {len(opt['regions'])} regions")

    r_pred = client.post("/api/predict", json={
        "suburb": "Richmond",
        "property_type": "h",
        "rooms": 3,
        "bathrooms": 2,
        "car_spaces": 2,
        "landsize": 320,
        "building_area": 155,
        "year_built": 1935,
        "interest_rate": 1.50,
        "max_budget": 1500000,
        "selected_model": "xgboost",
    })
    assert r_pred.status_code == 200
    pred = r_pred.get_json()
    assert "floor_to_land_ratio_pct" in pred["resolved_inputs"]
    assert "champion_test_r2_pct" in pred["resolved_inputs"]
    assert "shap_base_value" in pred
    assert "global_feature_importance" in pred and len(pred["global_feature_importance"]) >= 5

    # Verify TreeSHAP contributions sum with baseline to exact predicted price
    shap_sum = pred["shap_base_value"] + sum(d["shap_delta"] for d in pred["xai_drivers"])
    assert abs(shap_sum - pred["predicted_price"]) <= 15.0, (
        f"TreeSHAP reconciliation mismatch: {shap_sum} vs {pred['predicted_price']}"
    )

    # Verify TreeSHAP percentages change when inputs change (not static weights!)
    r_pred_unit = client.post("/api/predict", json={
        "suburb": "South Yarra",
        "property_type": "u",
        "rooms": 2,
        "bathrooms": 1,
        "car_spaces": 1,
        "landsize": 120,
        "building_area": 75,
        "year_built": 2005,
        "interest_rate": 1.50,
        "selected_model": "xgboost",
    })
    assert r_pred_unit.status_code == 200
    pred_unit = r_pred_unit.get_json()
    shares_house = {d["factor"]: d["share_pct"] for d in pred["xai_drivers"]}
    shares_unit = {d["factor"]: d["share_pct"] for d in pred_unit["xai_drivers"]}
    assert shares_house != shares_unit, f"Expected dynamic SHAP shares, got identical {shares_house}"

    # Verify interest rate is a display-only scenario input and does NOT alter model predictions
    r_pred_rate = client.post("/api/predict", json={
        "suburb": "Richmond",
        "property_type": "h",
        "rooms": 3,
        "bathrooms": 2,
        "car_spaces": 2,
        "landsize": 320,
        "building_area": 155,
        "year_built": 1935,
        "interest_rate": 4.10,
        "selected_model": "xgboost",
    })
    pred_rate = r_pred_rate.get_json()
    assert pred_rate["predicted_price"] == pred["predicted_price"], (
        f"Interest rate scenario should not alter model prediction: {pred_rate['predicted_price']} vs {pred['predicted_price']}"
    )
    print(
        f"[PASS] POST /api/predict -> Predicted Price: ${pred['predicted_price']:,.0f}, "
        f"SHAP Base: ${pred['shap_base_value']:,.0f}, Dynamic SHAP Shares Verified, "
        f"Comparables: {len(pred['comparable_properties'])}"
    )

    # Default /api/dashboard uses Holdout Test split only (5,439 sales)
    r_dash_test = client.get("/api/dashboard")
    assert r_dash_test.status_code == 200
    dash_test = r_dash_test.get_json()
    assert dash_test["active_split"] == "TEST"
    assert dash_test["kpis"]["total_sales"] == 5439, f"Expected 5439 Holdout Test sales, got {dash_test['kpis']['total_sales']}"
    assert dash_test["kpis"]["unseen_filtered_count"] == 7607
    assert len(dash_test["scatter_points"]) > 0
    assert len(dash_test["error_by_type_rooms"]) > 0
    assert len(dash_test["unseen_predictions"]) > 0
    assert "SuburbTypeMedianPrice" in dash_test["unseen_predictions"][0]
    distinct_unseen_suburbs = {r["Suburb"] for r in dash_test["unseen_predictions"]}
    assert len(distinct_unseen_suburbs) >= 10, f"Expected diverse suburbs, got {distinct_unseen_suburbs}"
    print(
        f"[PASS] GET /api/dashboard (Default=TEST) -> Holdout Sales: {dash_test['kpis']['total_sales']:,}, "
        f"Actual: ${dash_test['kpis']['avg_actual_price']:,.0f}, "
        f"Predicted: ${dash_test['kpis']['avg_predicted_price']:,.0f}, "
        f"Holdout MAE: ${dash_test['kpis']['mae_slice']:,.0f}, "
        f"Unseen Matched: {dash_test['kpis']['unseen_filtered_count']:,} ({len(distinct_unseen_suburbs)} distinct suburbs)"
    )

    # Also verify split=ALL (27,192 out-of-sample sales) and split=TRAIN (21,753 5-fold OOF sales)
    r_dash_all = client.get("/api/dashboard?split=ALL")
    assert r_dash_all.status_code == 200
    dash_all = r_dash_all.get_json()
    assert dash_all["kpis"]["total_sales"] == 27192
    r_dash_train = client.get("/api/dashboard?split=TRAIN")
    assert r_dash_train.status_code == 200
    dash_train = r_dash_train.get_json()
    assert dash_train["kpis"]["total_sales"] == 21753
    print(
        f"[PASS] GET /api/dashboard Split Toggle -> ALL: {dash_all['kpis']['total_sales']:,} sales, "
        f"TRAIN (5-Fold OOF): {dash_train['kpis']['total_sales']:,} sales"
    )

    r_dash = client.get("/api/dashboard?region=Southern+Metropolitan&property_type=h")
    assert r_dash.status_code == 200
    dash = r_dash.get_json()
    assert dash["kpis"]["unseen_filtered_count"] < 7607
    print(
        f"[PASS] GET /api/dashboard (Filtered) -> Sales: {dash['kpis']['total_sales']:,}, "
        f"Unseen Matched: {dash['kpis']['unseen_filtered_count']:,}"
    )

    r_dash_desc = client.get("/api/dashboard?unseen_sort=price_desc")
    assert r_dash_desc.status_code == 200
    unseen_desc = r_dash_desc.get_json()["unseen_predictions"]
    prices_desc = [r["PredictedPrice_XGB"] for r in unseen_desc]
    assert prices_desc == sorted(prices_desc, reverse=True), f"Expected descending prices, got {prices_desc}"
    print(f"[PASS] GET /api/dashboard?unseen_sort=price_desc -> Top Unseen Price: ${prices_desc[0]:,.0f} ({unseen_desc[0]['Suburb']})")

    r_abi = client.post("/api/abi/optimize", json={
        "budget": 1350000,
        "strategy": "balanced",
        "risk_tolerance": "moderate",
        "interest_rate": 1.50,
        "region": "ALL",
        "min_rooms": 2,
    })
    assert r_abi.status_code == 200
    abi_res = r_abi.get_json()
    print(f"[PASS] POST /api/abi/optimize -> Top Recommendations: {len(abi_res['recommendations'])}, Top ABI Score: {abi_res['recommendations'][0]['abi_utility_score']}")

    r_adapt = client.post("/api/abi/adapt", json={
        "suburb": "Richmond",
        "property_type": "h",
        "rooms": 3,
        "observed_price": 1380000,
    })
    assert r_adapt.status_code == 200
    adapt_res = r_adapt.get_json()
    print(f"[PASS] POST /api/abi/adapt -> Error Reduction: ${adapt_res['error_reduction']:,.0f}, Calibration Multiplier: {adapt_res['new_calibration']}")

    r_chat = client.post("/api/agent/chat", json={
        "message": "I want a 3 bedroom house in Richmond with 2 bathrooms under $1.4M"
    })
    assert r_chat.status_code == 200
    chat_res = r_chat.get_json()
    print(f"[PASS] POST /api/agent/chat -> Intent: {chat_res['intent']}, Database Matches: {len(chat_res['matching_properties'])}")

    r_etl = client.get("/api/etl?table=Fact_PropertySale")
    assert r_etl.status_code == 200
    etl_json = r_etl.get_json()
    dq_audit = etl_json["etl_metadata"].get("data_quality_audit", {})
    assert "duplicates_audit" in dq_audit
    assert "missing_values_by_column" in dq_audit and len(dq_audit["missing_values_by_column"]) == 21
    assert "outlier_rules_applied" in dq_audit and len(dq_audit["outlier_rules_applied"]) == 10
    assert "rows_dropped_and_separated" in dq_audit and len(dq_audit["rows_dropped_and_separated"]) == 8
    assert "encoding_and_scaling_scheme" in dq_audit and len(dq_audit["encoding_and_scaling_scheme"]) == 5

    r_etl_unseen = client.get("/api/etl?table=Fact_UnseenProperty")
    assert r_etl_unseen.status_code == 200
    assert len(r_etl_unseen.get_json()["sample_rows"]) > 0
    print(f"[PASS] GET /api/etl -> Fact_PropertySale ({len(etl_json['sample_rows'])} rows), Fact_UnseenProperty ({len(r_etl_unseen.get_json()['sample_rows'])} rows), Data Quality Audit Verified (21 cols, 10 outlier rules)")

    r_dq_csv = client.get("/api/etl/quality-report?format=csv")
    assert r_dq_csv.status_code == 200
    assert "text/csv" in r_dq_csv.content_type
    assert "1_Row_Disposition_and_Deduplication" in r_dq_csv.get_data(as_text=True)

    r_dq_json = client.get("/api/etl/quality-report?format=json")
    assert r_dq_json.status_code == 200
    assert "application/json" in r_dq_json.content_type
    print("[PASS] GET /api/etl/quality-report -> CSV & JSON downloadable reports verified")

    r_ml = client.get("/api/ml")
    assert r_ml.status_code == 200
    ml_json = r_ml.get_json()
    assert "conformal_intervals" in ml_json
    assert "residual_analysis" in ml_json
    assert len(ml_json["residual_analysis"]["by_price_band"]) == 5
    assert "cv_metrics" in ml_json["models"]["xgboost"]
    assert "MAPE" in ml_json["models"]["xgboost"]["test_metrics"]
    assert "tooltip" in pred["confidence_interval_95"]
    print(
        f"[PASS] GET /api/ml -> Champion Model: {ml_json['champion_model_name']}, "
        f"Holdout R²: {ml_json['models']['xgboost']['test_metrics']['R2']*100:.2f}%, "
        f"5-Fold CV R²: {ml_json['models']['xgboost']['cv_metrics']['R2_mean']*100:.2f}% ± {ml_json['models']['xgboost']['cv_metrics']['R2_std']*100:.2f}%, "
        f"Conformal Coverage: {ml_json['conformal_intervals']['empirical_test_coverage_95_pct']}%"
    )
    print("\nALL PLATFORM ENDPOINT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()

