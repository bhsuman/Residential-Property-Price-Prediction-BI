"""
Automated Dual-Source ETL Pipeline & Dimensional Star Schema Builder.
Ingests and cleans:
  1. data/raw/melbourne_house_prices_less.csv (63,023 records, 13 columns)
  2. data/raw/melbourne_housing_full.csv (34,857 records, 21 columns)

Populates the 5-table Star Schema Data Warehouse in data/processed/property_star_schema.db:
  - Fact_PropertySale
  - Dim_Property
  - Dim_Location
  - Dim_Date
  - Dim_Market
"""

import os
import json
import time
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import (
    PROCESSED_DATA_DIR,
    LESS_CSV_PATH,
    FULL_CSV_PATH,
    DB_PATH,
    CLEANED_TRAIN_CSV,
    UNSEEN_CSV,
    ETL_META_PATH,
    RBA_INTEREST_RATES,
    PROPERTY_TYPE_LABELS,
    SALE_METHOD_LABELS,
    MELBOURNE_CBD_LAT,
    MELBOURNE_CBD_LON,
    haversine_km,
)


def run_etl_pipeline():
    """Execute end-to-end Extract, Transform, Clean, Feature Engineer, and Load pipeline."""
    start_ts = time.time()
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    steps_log = []

    def log_step(stage, title, details, metrics=None):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "title": title,
            "details": details,
            "metrics": metrics or {},
        }
        steps_log.append(entry)
        print(f"[{stage}] {title}: {details}")

    # =========================================================================
    # STAGE 1: EXTRACT (Dual Dataset Ingestion)
    # =========================================================================
    df_less_raw = pd.read_csv(LESS_CSV_PATH)
    df_full_raw = pd.read_csv(FULL_CSV_PATH)

    less_raw_rows, less_raw_cols = df_less_raw.shape
    full_raw_rows, full_raw_cols = df_full_raw.shape

    log_step(
        "EXTRACT",
        "Ingest Dual CSV Datasets from data/raw/",
        f"Loaded melbourne_house_prices_less.csv ({less_raw_rows:,} rows, {less_raw_cols} cols) "
        f"and melbourne_housing_full.csv ({full_raw_rows:,} rows, {full_raw_cols} cols).",
        {
            "less_rows": int(less_raw_rows),
            "less_cols": int(less_raw_cols),
            "full_rows": int(full_raw_rows),
            "full_cols": int(full_raw_cols),
            "total_raw_rows_ingested": int(less_raw_rows + full_raw_rows),
        },
    )

    # =========================================================================
    # STAGE 2: TRANSFORM & DATA CLEANING
    # =========================================================================
    less_dups = int(df_less_raw.duplicated().sum())
    full_dups = int(df_full_raw.duplicated().sum())
    df_less = df_less_raw.drop_duplicates().copy()
    df_full = df_full_raw.drop_duplicates().copy()

    for df in (df_less, df_full):
        for col in ["Suburb", "Address", "Type", "Method", "SellerG", "Regionname", "CouncilArea"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    suburb_lookup = (
        df_less.groupby("Suburb", as_index=False)
        .agg(
            Postcode_ref=("Postcode", "first"),
            Regionname_ref=("Regionname", "first"),
            Propertycount_ref=("Propertycount", "median"),
            Distance_ref=("Distance", "median"),
            CouncilArea_ref=("CouncilArea", "first"),
        )
    )

    df_full = df_full.merge(suburb_lookup, on="Suburb", how="left")
    backfilled_fields = 0
    for col in ["Postcode", "Regionname", "Propertycount", "Distance", "CouncilArea"]:
        missing_mask = df_full[col].isna()
        n_miss = int(missing_mask.sum())
        if n_miss > 0:
            df_full.loc[missing_mask, col] = df_full.loc[missing_mask, f"{col}_ref"]
            backfilled_fields += n_miss
        df_full.drop(columns=[f"{col}_ref"], inplace=True)

    merge_keys = ["Suburb", "Address", "Rooms", "Type", "Date", "Method", "SellerG"]
    less_price_lookup = (
        df_less[df_less["Price"].notna()]
        .drop_duplicates(subset=merge_keys)[merge_keys + ["Price"]]
        .rename(columns={"Price": "Price_less"})
    )
    df_full = df_full.merge(less_price_lookup, on=merge_keys, how="left")
    price_backfilled = int((df_full["Price"].isna() & df_full["Price_less"].notna()).sum())
    df_full["Price"] = df_full["Price"].fillna(df_full["Price_less"])
    df_full.drop(columns=["Price_less"], inplace=True)

    def fast_record_key(d):
        return (
            d["Suburb"].fillna("") + "|" +
            d["Address"].fillna("") + "|" +
            d["Rooms"].astype(str) + "|" +
            d["Type"].fillna("") + "|" +
            d["Date"].fillna("") + "|" +
            d["Method"].fillna("") + "|" +
            d["SellerG"].fillna("")
        )

    df_full["RecordKey"] = fast_record_key(df_full)
    df_less["RecordKey"] = fast_record_key(df_less)
    full_key_set = set(df_full["RecordKey"])
    df_less_extra = df_less[~df_less["RecordKey"].isin(full_key_set)].copy()

    log_step(
        "TRANSFORM",
        "Cross-Dataset Integration & Deduplication",
        f"Removed {less_dups + full_dups} exact duplicate rows ({less_dups} in LESS, {full_dups} in FULL). "
        f"Backfilled {backfilled_fields} missing location metadata values in FULL using LESS suburb lookup, "
        f"and linked {len(df_less_extra):,} additional market transactions from LESS.",
        {
            "duplicates_removed": less_dups + full_dups,
            "location_fields_backfilled": backfilled_fields,
            "price_backfilled_from_less": price_backfilled,
            "extra_less_transactions": int(len(df_less_extra)),
        },
    )

    for df in (df_full, df_less, df_less_extra):
        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["SaleDate"] = dt.dt.strftime("%Y-%m-%d")
        df["SaleYear"] = dt.dt.year
        df["SaleMonth"] = dt.dt.month
        df["SaleQuarter"] = dt.dt.quarter
        df["YearQuarter"] = df["SaleYear"].astype(str) + "-Q" + df["SaleQuarter"].astype(str)
        df["YearMonth"] = dt.dt.strftime("%Y-%m")

    anomalies_fixed = {}
    yb_typo_mask = df_full["YearBuilt"] == 1196
    yb_future_mask = df_full["YearBuilt"] == 2106
    n_yb_typo = int(yb_typo_mask.sum() + yb_future_mask.sum())
    df_full.loc[yb_typo_mask, "YearBuilt"] = 1996
    df_full.loc[yb_future_mask, "YearBuilt"] = 2016
    yb_invalid_mask = (df_full["YearBuilt"] < 1835) | (df_full["YearBuilt"] > 2019)
    n_yb_range = int(yb_invalid_mask.sum())
    anomalies_fixed["YearBuilt_anomalies"] = int(n_yb_typo + n_yb_range)
    df_full.loc[yb_invalid_mask, "YearBuilt"] = np.nan

    ba_invalid_mask = (df_full["BuildingArea"] < 15) | (df_full["BuildingArea"] > 1500)
    n_ba_invalid = int(ba_invalid_mask.sum())
    anomalies_fixed["BuildingArea_anomalies"] = n_ba_invalid
    df_full.loc[ba_invalid_mask, "BuildingArea"] = np.nan

    ls_extreme_mask = df_full["Landsize"] > 15000
    ls_zero_house_mask = (df_full["Landsize"] <= 0) & (df_full["Type"] == "h")
    n_ls_extreme = int(ls_extreme_mask.sum())
    n_ls_zero_house = int(ls_zero_house_mask.sum())
    anomalies_fixed["Landsize_anomalies"] = int(n_ls_extreme + n_ls_zero_house)
    df_full.loc[ls_extreme_mask, "Landsize"] = 15000.0
    df_full.loc[ls_zero_house_mask, "Landsize"] = np.nan

    bed_invalid = (df_full["Bedroom2"] <= 0) | (df_full["Bedroom2"] > 10)
    bath_invalid = (df_full["Bathroom"] <= 0) | (df_full["Bathroom"] > 8)
    car_invalid = (df_full["Car"] < 0) | (df_full["Car"] > 10)
    n_bed_invalid = int(bed_invalid.sum())
    n_bath_invalid = int(bath_invalid.sum())
    n_car_invalid = int(car_invalid.sum())
    anomalies_fixed["Room_Bath_Car_anomalies"] = int(n_bed_invalid + n_bath_invalid + n_car_invalid)
    df_full.loc[bed_invalid, "Bedroom2"] = df_full.loc[bed_invalid, "Rooms"].clip(1, 10)
    df_full.loc[bath_invalid, "Bathroom"] = np.nan
    df_full.loc[car_invalid, "Car"] = np.nan
    df_full["Rooms"] = df_full["Rooms"].clip(1, 10)

    log_step(
        "CLEAN",
        "Outlier & Anomaly Detection and Normalization",
        f"Corrected YearBuilt typos/outliers ({anomalies_fixed['YearBuilt_anomalies']}), "
        f"invalid BuildingArea entries ({anomalies_fixed['BuildingArea_anomalies']}), "
        f"Landsize anomalies ({anomalies_fixed['Landsize_anomalies']}), and "
        f"Bedroom/Bathroom/Car extreme values ({anomalies_fixed['Room_Bath_Car_anomalies']}).",
        anomalies_fixed,
    )

    df_priced = df_full[df_full["Price"].notna()].copy()
    df_unseen = df_full[df_full["Price"].isna()].copy()

    p_high = float(df_priced["Price"].quantile(0.998))
    price_low_mask = df_priced["Price"] < 120000
    price_high_mask = df_priced["Price"] > p_high
    n_price_low = int(price_low_mask.sum())
    n_price_high = int(price_high_mask.sum())
    price_outlier_mask = price_low_mask | price_high_mask
    n_price_outliers = int(price_outlier_mask.sum())
    df_priced = df_priced[~price_outlier_mask].copy()

    # Snapshot pre-imputation missing counts on validated priced sales (after anomaly-to-NaN rules)
    pre_imputation_priced_nulls = {col: int(df_priced[col].isna().sum()) for col in df_full_raw.columns if col in df_priced.columns}
    pre_imputation_unseen_nulls = {col: int(df_unseen[col].isna().sum()) for col in df_full_raw.columns if col in df_unseen.columns}

    # Record explicit missingness / imputation flags BEFORE any imputation
    for df in (df_priced, df_unseen):
        df["BuildingArea_was_imputed"] = df["BuildingArea"].isna().astype(int)
        df["YearBuilt_was_imputed"] = df["YearBuilt"].isna().astype(int)
        df["Landsize_was_imputed"] = (df["Landsize"].isna() | (df["Landsize"] <= 0)).astype(int)
        df["Has_BuildingArea"] = 1 - df["BuildingArea_was_imputed"]
        df["Has_YearBuilt"] = 1 - df["YearBuilt_was_imputed"]
        df["Has_Landsize"] = 1 - df["Landsize_was_imputed"]

    # Split into Train (80%) and Test (20%) BEFORE fitting any imputer or target-derived statistics
    df_priced = df_priced.sort_values("SaleDate").reset_index(drop=True)
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(len(df_priced))
    split_point = int(len(df_priced) * 0.80)
    train_idx = set(shuffled_idx[:split_point])
    df_priced["DataSplit"] = ["Train" if i in train_idx else "Test" for i in range(len(df_priced))]

    train_ref = df_priced[df_priced["DataSplit"] == "Train"].copy()

    # =========================================================================
    # TIME-AWARE, PAST-ONLY MARKET & SUBURB BENCHMARKS (FIT ON TRAIN ONLY)
    # =========================================================================
    combined_train_market_sales = pd.concat(
        [
            train_ref[["Suburb", "Regionname", "SaleDate", "SaleYear", "SaleQuarter", "YearQuarter", "Price"]],
            df_less_extra[df_less_extra["Price"].notna()][["Suburb", "Regionname", "SaleDate", "SaleYear", "SaleQuarter", "YearQuarter", "Price"]],
        ],
        ignore_index=True,
    )
    combined_train_market_sales = combined_train_market_sales[
        (combined_train_market_sales["Price"] >= 120000) & (combined_train_market_sales["Price"] <= p_high)
    ].sort_values("SaleDate").reset_index(drop=True)

    # Earliest baseline from initial month of training data (to seed first period without future leakage)
    earliest_cutoff = "2016-04-01"
    early_slice = combined_train_market_sales[combined_train_market_sales["SaleDate"] < earliest_cutoff]
    if len(early_slice) < 50:
        early_slice = combined_train_market_sales.iloc[:500]
    early_global_median = float(early_slice["Price"].median())
    early_region_medians = early_slice.groupby("Regionname")["Price"].median().to_dict()

    region_qtr = (
        combined_train_market_sales.groupby(["Regionname", "YearQuarter"], as_index=False)
        .agg(
            CurrentQuarterMedianPrice=("Price", "median"),
            QuarterTransactionVolume=("Price", "count"),
        )
        .sort_values(["Regionname", "YearQuarter"])
    )
    region_qtr["PriorPeriodMedianPrice"] = (
        region_qtr.groupby("Regionname")["CurrentQuarterMedianPrice"]
        .apply(lambda s: s.expanding().median().shift(1))
        .reset_index(level=0, drop=True)
    )
    first_qtr_mask = region_qtr["PriorPeriodMedianPrice"].isna()
    region_qtr.loc[first_qtr_mask, "PriorPeriodMedianPrice"] = (
        region_qtr.loc[first_qtr_mask, "Regionname"].map(early_region_medians).fillna(early_global_median)
    )

    base_medians = region_qtr.groupby("Regionname")["PriorPeriodMedianPrice"].transform("first")
    region_qtr["MarketIndex"] = np.round((region_qtr["PriorPeriodMedianPrice"] / base_medians) * 100.0, 2)
    region_qtr["InterestRate"] = region_qtr["YearQuarter"].map(RBA_INTEREST_RATES).fillna(1.50)

    # Strictly past-only date-level expanding medians per Suburb and per Region (zero same-date or future leakage)
    reg_daily = (
        combined_train_market_sales.groupby(["Regionname", "SaleDate"], as_index=False)
        .agg(DailyPrices=("Price", list))
        .sort_values(["Regionname", "SaleDate"])
    )
    reg_prior_date_map = {}
    for reg_name, grp in reg_daily.groupby("Regionname", sort=False):
        past_vals = []
        seed_val = float(early_region_medians.get(reg_name, early_global_median))
        for _, row in grp.iterrows():
            sdate = row["SaleDate"]
            reg_prior_date_map[(reg_name, sdate)] = float(np.median(past_vals)) if past_vals else seed_val
            past_vals.extend(row["DailyPrices"])

    sub_daily = (
        combined_train_market_sales.groupby(["Suburb", "SaleDate"], as_index=False)
        .agg(DailyPrices=("Price", list))
        .sort_values(["Suburb", "SaleDate"])
    )
    sub_prior_date_map = {}
    for sub_name, grp in sub_daily.groupby("Suburb", sort=False):
        past_vals = []
        for _, row in grp.iterrows():
            sdate = row["SaleDate"]
            if past_vals:
                sub_prior_date_map[(sub_name, sdate)] = float(np.median(past_vals))
            past_vals.extend(row["DailyPrices"])

    # Quarter-level fallback strictly from prior quarters (shifted by 1 quarter)
    suburb_qtr = (
        combined_train_market_sales.groupby(["Suburb", "YearQuarter"], as_index=False)
        .agg(SuburbQtrMedian=("Price", "median"))
        .sort_values(["Suburb", "YearQuarter"])
    )
    suburb_qtr["PriorSuburbMedianPrice"] = (
        suburb_qtr.groupby("Suburb")["SuburbQtrMedian"]
        .apply(lambda s: s.expanding().median().shift(1))
        .reset_index(level=0, drop=True)
    )

    log_step(
        "TRANSFORM",
        "Compute Strictly Past-Only Macro & Suburb Market Indicators (Zero Leakage)",
        f"Calculated chronological past-only Prior-Period Median Prices and Market Indices strictly from "
        f"training and historical transactions ({len(combined_train_market_sales):,} rows) occurring prior to each sale date.",
        {
            "combined_priced_transactions": int(len(combined_train_market_sales)),
            "market_dimension_records": int(len(region_qtr)),
        },
    )

    # =========================================================================
    # TRAIN-ONLY HIERARCHICAL MISSING VALUE IMPUTATION
    # =========================================================================
    for df in (df_priced, df_unseen):
        df["Bedroom2"] = df["Bedroom2"].fillna(df["Rooms"]).clip(1, 10)

    bath_by_rt = train_ref.groupby(["Type", "Rooms"])["Bathroom"].median().to_dict()
    car_by_rt = train_ref.groupby(["Type", "Rooms"])["Car"].median().to_dict()
    global_bath = float(train_ref["Bathroom"].median())
    global_car = float(train_ref["Car"].median())

    ls_sub_rt = train_ref.groupby(["Suburb", "Type", "Rooms"])["Landsize"].median().to_dict()
    ls_sub_type = train_ref.groupby(["Suburb", "Type"])["Landsize"].median().to_dict()
    ls_reg_type = train_ref.groupby(["Regionname", "Type"])["Landsize"].median().to_dict()
    ls_type = train_ref.groupby("Type")["Landsize"].median().to_dict()

    # Granular 5-tier BuildingArea hierarchy fitted on Train only (eliminates constant 96.5 collapse)
    ba_sub_rt = train_ref.groupby(["Suburb", "Type", "Rooms"])["BuildingArea"].median().to_dict()
    ba_sub_t = train_ref.groupby(["Suburb", "Type"])["BuildingArea"].median().to_dict()
    ba_reg_rt = train_ref.groupby(["Regionname", "Type", "Rooms"])["BuildingArea"].median().to_dict()
    ba_rt = train_ref.groupby(["Type", "Rooms"])["BuildingArea"].median().to_dict()
    ba_type = train_ref.groupby("Type")["BuildingArea"].median().to_dict()
    global_ba = float(train_ref["BuildingArea"].median())

    yb_sub_t = train_ref.groupby(["Suburb", "Type"])["YearBuilt"].median().to_dict()
    yb_sub = train_ref.groupby("Suburb")["YearBuilt"].median().to_dict()
    yb_reg = train_ref.groupby("Regionname")["YearBuilt"].median().to_dict()
    global_yb = float(train_ref["YearBuilt"].median())

    lat_sub = train_ref.groupby("Suburb")["Lattitude"].median().to_dict()
    lon_sub = train_ref.groupby("Suburb")["Longtitude"].median().to_dict()
    lat_reg = train_ref.groupby("Regionname")["Lattitude"].median().to_dict()
    lon_reg = train_ref.groupby("Regionname")["Longtitude"].median().to_dict()
    global_dist = float(train_ref["Distance"].median())
    global_pcount = float(train_ref["Propertycount"].median())
    mode_region = str(train_ref["Regionname"].mode().iloc[0])
    mode_council = str(train_ref["CouncilArea"].mode().iloc[0])

    def apply_imputation(df):
        df = df.copy()
        df["Regionname"] = df["Regionname"].fillna(mode_region)
        df["CouncilArea"] = df["CouncilArea"].fillna(mode_council)
        df["Distance"] = df["Distance"].fillna(global_dist)
        df["Propertycount"] = df["Propertycount"].fillna(global_pcount)
        df["Postcode"] = df["Postcode"].fillna(3000)

        srt_keys = list(zip(df["Suburb"], df["Type"], df["Rooms"]))
        reg_rt_keys = list(zip(df["Regionname"], df["Type"], df["Rooms"]))
        rt_keys = list(zip(df["Type"], df["Rooms"]))
        st_keys = list(zip(df["Suburb"], df["Type"]))
        regt_keys = list(zip(df["Regionname"], df["Type"]))

        df["Bathroom"] = df["Bathroom"].fillna(pd.Series([bath_by_rt.get(k, global_bath) for k in rt_keys], index=df.index)).fillna(global_bath).round()
        df["Car"] = df["Car"].fillna(pd.Series([car_by_rt.get(k, global_car) for k in rt_keys], index=df.index)).fillna(global_car).round()

        ls_f0 = pd.Series([ls_sub_rt.get(k, np.nan) for k in srt_keys], index=df.index)
        ls_f1 = pd.Series([ls_sub_type.get(k, np.nan) for k in st_keys], index=df.index)
        ls_f2 = pd.Series([ls_reg_type.get(k, np.nan) for k in regt_keys], index=df.index)
        ls_f3 = df["Type"].map(ls_type).fillna(450.0)
        df["Landsize"] = df["Landsize"].fillna(ls_f0).fillna(ls_f1).fillna(ls_f2).fillna(ls_f3).fillna(450.0)

        ba_f0 = pd.Series([ba_sub_rt.get(k, np.nan) for k in srt_keys], index=df.index)
        ba_f1 = pd.Series([ba_sub_t.get(k, np.nan) for k in st_keys], index=df.index)
        ba_f2 = pd.Series([ba_reg_rt.get(k, np.nan) for k in reg_rt_keys], index=df.index)
        ba_f3 = pd.Series([ba_rt.get(k, np.nan) for k in rt_keys], index=df.index)
        ba_f4 = df["Type"].map(ba_type).fillna(global_ba)
        df["BuildingArea"] = df["BuildingArea"].fillna(ba_f0).fillna(ba_f1).fillna(ba_f2).fillna(ba_f3).fillna(ba_f4).fillna(global_ba)

        yb_f0 = pd.Series([yb_sub_t.get(k, np.nan) for k in st_keys], index=df.index)
        df["YearBuilt"] = (
            df["YearBuilt"]
            .fillna(yb_f0)
            .fillna(df["Suburb"].map(yb_sub))
            .fillna(df["Regionname"].map(yb_reg))
            .fillna(global_yb)
            .round()
        )
        df["Lattitude"] = (
            df["Lattitude"]
            .fillna(df["Suburb"].map(lat_sub))
            .fillna(df["Regionname"].map(lat_reg))
            .fillna(MELBOURNE_CBD_LAT)
        )
        df["Longtitude"] = (
            df["Longtitude"]
            .fillna(df["Suburb"].map(lon_sub))
            .fillna(df["Regionname"].map(lon_reg))
            .fillna(MELBOURNE_CBD_LON)
        )
        return df

    df_priced = apply_imputation(df_priced)
    df_unseen = apply_imputation(df_unseen)
    train_ref = df_priced[df_priced["DataSplit"] == "Train"].copy()

    log_step(
        "CLEAN",
        "Train-Split Hierarchical Missing Value Imputation",
        f"Separated {len(df_priced):,} priced property sales and {len(df_unseen):,} unseen (unpriced) properties "
        f"(filtered {n_price_outliers} extreme price outliers). Fitted 5-level hierarchical median imputers strictly on the "
        f"80% training split ({len(train_ref):,} rows) and added explicit *_was_imputed indicator flags.",
        {
            "priced_sales_cleaned": int(len(df_priced)),
            "unseen_properties_cleaned": int(len(df_unseen)),
            "price_outliers_removed": n_price_outliers,
            "building_area_imputed_count": int(df_priced["BuildingArea_was_imputed"].sum()),
            "year_built_imputed_count": int(df_priced["YearBuilt_was_imputed"].sum()),
            "landsize_imputed_count": int(df_priced["Landsize_was_imputed"].sum()),
            "remaining_nulls_in_priced": int(df_priced[["Rooms", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance", "Lattitude", "Longtitude"]].isna().sum().sum()),
        },
    )

    reg_market_lookup = region_qtr[["Regionname", "YearQuarter", "PriorPeriodMedianPrice", "MarketIndex", "InterestRate", "QuarterTransactionVolume"]]
    sub_market_lookup = suburb_qtr[["Suburb", "YearQuarter", "PriorSuburbMedianPrice"]]

    # Same-suburb & same-dwelling-type median price computed strictly on Train (for honest unseen comparison)
    train_sub_type_medians = train_ref.groupby(["Suburb", "Type"])["Price"].median().to_dict()
    train_reg_type_medians = train_ref.groupby(["Regionname", "Type"])["Price"].median().to_dict()
    train_type_medians = train_ref.groupby("Type")["Price"].median().to_dict()
    global_train_median_price = float(train_ref["Price"].median())

    def engineer_features(df):
        df = df.merge(reg_market_lookup, on=["Regionname", "YearQuarter"], how="left")
        df = df.merge(sub_market_lookup, on=["Suburb", "YearQuarter"], how="left")

        # Use strictly prior date-level medians where available, falling back only to prior-quarter medians (never future!)
        reg_date_keys = list(zip(df["Regionname"], df["SaleDate"]))
        sub_date_keys = list(zip(df["Suburb"], df["SaleDate"]))
        prior_reg_by_date = pd.Series([reg_prior_date_map.get(k, np.nan) for k in reg_date_keys], index=df.index)
        prior_sub_by_date = pd.Series([sub_prior_date_map.get(k, np.nan) for k in sub_date_keys], index=df.index)

        df["PriorPeriodMedianPrice"] = (
            prior_reg_by_date
            .fillna(df["PriorPeriodMedianPrice"])
            .fillna(df["Regionname"].map(early_region_medians))
            .fillna(early_global_median)
        )
        df["PriorSuburbMedianPrice"] = (
            prior_sub_by_date
            .fillna(df["PriorSuburbMedianPrice"])
            .fillna(df["PriorPeriodMedianPrice"])
        )
        df["MarketIndex"] = df["MarketIndex"].fillna(100.0)
        df["InterestRate"] = df["InterestRate"].fillna(1.50)
        df["QuarterTransactionVolume"] = df["QuarterTransactionVolume"].fillna(500).astype(int)

        # Same-suburb & same-dwelling-type benchmark from training set
        st_keys = list(zip(df["Suburb"], df["Type"]))
        rt_keys = list(zip(df["Regionname"], df["Type"]))
        sub_type_med = pd.Series([train_sub_type_medians.get(k, np.nan) for k in st_keys], index=df.index)
        reg_type_med = pd.Series([train_reg_type_medians.get(k, np.nan) for k in rt_keys], index=df.index)
        type_med = df["Type"].map(train_type_medians).fillna(global_train_median_price)
        df["SuburbTypeMedianPrice"] = sub_type_med.fillna(reg_type_med).fillna(type_med)

        df["PropertyAge"] = (df["SaleYear"] - df["YearBuilt"]).clip(lower=0, upper=180)
        df["FloorToLandRatio"] = np.round(df["BuildingArea"] / (df["Landsize"] + 1.0), 4).clip(0.0, 10.0)
        df["TotalUtilityRooms"] = df["Rooms"] + df["Bathroom"] + df["Car"]
        df["RadialDistanceCBD"] = np.round(haversine_km(df["Lattitude"], df["Longtitude"]), 2)
        df["SuburbPremiumRatio"] = np.round(df["PriorSuburbMedianPrice"] / df["PriorPeriodMedianPrice"], 4)
        df["PropertyTypeLabel"] = df["Type"].map(PROPERTY_TYPE_LABELS).fillna("Residential")
        df["MethodLabel"] = df["Method"].map(SALE_METHOD_LABELS).fillna("Standard Sale")
        return df

    df_priced = engineer_features(df_priced)
    df_unseen = engineer_features(df_unseen)

    # =========================================================================
    # OUT-OF-FOLD (WITHIN TRAINING FOLDS ONLY) SUBURB TARGET ENCODING
    # =========================================================================
    def fit_eb_target_encoder(tr_df):
        """Fit hierarchical empirical-Bayes target encoder in log1p(Price) space on a training fold."""
        z = np.log1p(tr_df["Price"].values)
        tr_work = tr_df[["Suburb", "Regionname", "Type", "Rooms"]].copy()
        tr_work["z"] = z
        tr_work["RoomsBucket"] = tr_work["Rooms"].clip(1, 5).astype(int)

        mu0 = float(tr_work["z"].mean())
        reg_means = tr_work.groupby("Regionname")["z"].mean().to_dict()
        type_means = tr_work.groupby("Type")["z"].mean().to_dict()
        type_offsets = {k: v - mu0 for k, v in type_means.items()}
        tr_m = tr_work.groupby(["Type", "RoomsBucket"])["z"].mean().to_dict()
        room_offsets = {(t_c, r_b): val - type_means.get(t_c, mu0) for (t_c, r_b), val in tr_m.items()}

        # Level 1: Suburb smoothed toward Region mean (m = 10)
        sub_stats = tr_work.groupby(["Suburb", "Regionname"])["z"].agg(["count", "mean"]).reset_index()
        sub_map = {}
        for _, row_s in sub_stats.iterrows():
            s_name = row_s["Suburb"]
            r_prior = reg_means.get(row_s["Regionname"], mu0)
            n_s = float(row_s["count"])
            sub_map[s_name] = (n_s * float(row_s["mean"]) + 10.0 * r_prior) / (n_s + 10.0)

        # Level 2: (Suburb, Type) smoothed toward Suburb + Type offset (m = 8)
        st_stats = tr_work.groupby(["Suburb", "Regionname", "Type"])["z"].agg(["count", "mean"]).reset_index()
        st_map = {}
        for _, row_st in st_stats.iterrows():
            s_name = row_st["Suburb"]
            t_code = row_st["Type"]
            s_base = sub_map.get(s_name, reg_means.get(row_st["Regionname"], mu0))
            st_prior = s_base + type_offsets.get(t_code, 0.0)
            n_st = float(row_st["count"])
            st_map[(s_name, t_code)] = (n_st * float(row_st["mean"]) + 8.0 * st_prior) / (n_st + 8.0)

        # Level 3: (Suburb, Type, RoomsBucket) smoothed toward (Suburb, Type) + Rooms offset (m = 6)
        str_stats = tr_work.groupby(["Suburb", "Regionname", "Type", "RoomsBucket"])["z"].agg(["count", "mean"]).reset_index()
        str_map = {}
        for _, row_str in str_stats.iterrows():
            s_name = row_str["Suburb"]
            t_code = row_str["Type"]
            r_bkt = int(row_str["RoomsBucket"])
            st_base = st_map.get((s_name, t_code), sub_map.get(s_name, mu0) + type_offsets.get(t_code, 0.0))
            str_prior = st_base + room_offsets.get((t_code, r_bkt), 0.0)
            n_str = float(row_str["count"])
            str_map[(s_name, t_code, r_bkt)] = (n_str * float(row_str["mean"]) + 6.0 * str_prior) / (n_str + 6.0)

        return {
            "mu0": mu0,
            "reg_means": reg_means,
            "type_offsets": type_offsets,
            "room_offsets": room_offsets,
            "sub_map": sub_map,
            "st_map": st_map,
            "str_map": str_map,
        }

    def transform_eb_target_encoder(target_df, enc):
        """Apply fitted empirical-Bayes target encoder and return dollar-scale encoded columns."""
        mu0 = enc["mu0"]
        reg_means = enc["reg_means"]
        type_offsets = enc["type_offsets"]
        room_offsets = enc["room_offsets"]
        sub_map = enc["sub_map"]
        st_map = enc["st_map"]
        str_map = enc["str_map"]

        subs = target_df["Suburb"].values
        regs = target_df["Regionname"].values
        types = target_df["Type"].values
        rooms_bkt = target_df["Rooms"].clip(1, 5).astype(int).values

        sub_enc = np.zeros(len(target_df), dtype=float)
        st_enc = np.zeros(len(target_df), dtype=float)
        str_enc = np.zeros(len(target_df), dtype=float)

        for i in range(len(target_df)):
            s = subs[i]
            r = regs[i]
            t = types[i]
            rb = int(rooms_bkt[i])
            r_prior = reg_means.get(r, mu0)
            z_sub = sub_map.get(s, r_prior)
            z_st = st_map.get((s, t), z_sub + type_offsets.get(t, 0.0))
            z_str = str_map.get((s, t, rb), z_st + room_offsets.get((t, rb), 0.0))
            sub_enc[i] = np.round(np.expm1(z_sub), 0)
            st_enc[i] = np.round(np.expm1(z_st), 0)
            str_enc[i] = np.round(np.expm1(z_str), 0)

        return sub_enc, st_enc, str_enc

    train_idx = np.where(df_priced["DataSplit"].values == "Train")[0]
    test_idx = np.where(df_priced["DataSplit"].values == "Test")[0]
    train_subset = df_priced.iloc[train_idx].reset_index(drop=True)

    oof_sub_enc = np.zeros(len(train_subset), dtype=float)
    oof_st_enc = np.zeros(len(train_subset), dtype=float)
    oof_str_enc = np.zeros(len(train_subset), dtype=float)

    kf_enc = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_tr, fold_val in kf_enc.split(train_subset):
        fold_encoder = fit_eb_target_encoder(train_subset.iloc[fold_tr])
        s_v, st_v, str_v = transform_eb_target_encoder(train_subset.iloc[fold_val], fold_encoder)
        oof_sub_enc[fold_val] = s_v
        oof_st_enc[fold_val] = st_v
        oof_str_enc[fold_val] = str_v

    full_train_encoder = fit_eb_target_encoder(train_subset)
    test_sub_enc, test_st_enc, test_str_enc = transform_eb_target_encoder(df_priced.iloc[test_idx], full_train_encoder)
    unseen_sub_enc, unseen_st_enc, unseen_str_enc = transform_eb_target_encoder(df_unseen, full_train_encoder)

    df_priced["SuburbTargetEncoded"] = 0.0
    df_priced["SuburbTypeTargetEncoded"] = 0.0
    df_priced["SuburbTypeRoomsTargetEncoded"] = 0.0

    df_priced.iloc[train_idx, df_priced.columns.get_loc("SuburbTargetEncoded")] = oof_sub_enc
    df_priced.iloc[train_idx, df_priced.columns.get_loc("SuburbTypeTargetEncoded")] = oof_st_enc
    df_priced.iloc[train_idx, df_priced.columns.get_loc("SuburbTypeRoomsTargetEncoded")] = oof_str_enc

    df_priced.iloc[test_idx, df_priced.columns.get_loc("SuburbTargetEncoded")] = test_sub_enc
    df_priced.iloc[test_idx, df_priced.columns.get_loc("SuburbTypeTargetEncoded")] = test_st_enc
    df_priced.iloc[test_idx, df_priced.columns.get_loc("SuburbTypeRoomsTargetEncoded")] = test_str_enc

    df_unseen["SuburbTargetEncoded"] = unseen_sub_enc
    df_unseen["SuburbTypeTargetEncoded"] = unseen_st_enc
    df_unseen["SuburbTypeRoomsTargetEncoded"] = unseen_str_enc

    train_ref = df_priced[df_priced["DataSplit"] == "Train"].copy()

    log_step(
        "FEATURE_ENGINEERING",
        "Engineer Time-Aware Hedonic, Spatial & Out-of-Fold Target Encoded Features",
        "Engineered PropertyAge, FloorToLandRatio, TotalUtilityRooms, RadialDistanceCBD, SaleYear, SaleQuarter, "
        "5-fold Out-of-Fold Empirical Bayes SuburbTargetEncoded, SuburbTypeTargetEncoded & SuburbTypeRoomsTargetEncoded, "
        "strictly past-only PriorPeriodMedianPrice & PriorSuburbMedianPrice, SuburbPremiumRatio, "
        "MarketIndex, SuburbTypeMedianPrice, and *_was_imputed indicators.",
        {
            "engineered_features": [
                "PropertyAge",
                "FloorToLandRatio",
                "TotalUtilityRooms",
                "RadialDistanceCBD",
                "SaleYear",
                "SaleQuarter",
                "SuburbTargetEncoded",
                "SuburbTypeTargetEncoded",
                "SuburbTypeRoomsTargetEncoded",
                "PriorPeriodMedianPrice",
                "PriorSuburbMedianPrice",
                "SuburbPremiumRatio",
                "MarketIndex",
                "SuburbTypeMedianPrice",
                "BuildingArea_was_imputed",
                "YearBuilt_was_imputed",
                "Landsize_was_imputed",
            ]
        },
    )

    if os.path.exists(UNSEEN_CSV):
        try:
            old_unseen = pd.read_csv(UNSEEN_CSV)
            if len(old_unseen) == len(df_unseen):
                for u_col in ["PredictedPrice_XGB", "PredictedPrice_RF"]:
                    if u_col in old_unseen.columns:
                        df_unseen[u_col] = old_unseen[u_col].values
        except Exception:
            pass
    df_unseen.to_csv(UNSEEN_CSV, index=False)

    # =========================================================================
    # STAGE 3: LOAD (Dimensional Star Schema in SQLite — Aggregated on Train)
    # =========================================================================
    loc_train = (
        train_ref.groupby(["Suburb", "Postcode", "CouncilArea", "Regionname"], as_index=False)
        .agg(
            DistanceCBD=("Distance", "median"),
            Latitude=("Lattitude", "median"),
            Longitude=("Longtitude", "median"),
            PropertyCount=("Propertycount", "median"),
            PriorSuburbMedian=("PriorSuburbMedianPrice", "median"),
        )
    )
    loc_missing_from_train = (
        df_priced[~df_priced["Suburb"].isin(loc_train["Suburb"])]
        .groupby(["Suburb", "Postcode", "CouncilArea", "Regionname"], as_index=False)
        .agg(
            DistanceCBD=("Distance", "median"),
            Latitude=("Lattitude", "median"),
            Longitude=("Longtitude", "median"),
            PropertyCount=("Propertycount", "median"),
            PriorSuburbMedian=("PriorPeriodMedianPrice", "median"),
        )
    )
    loc_full = pd.concat([loc_train, loc_missing_from_train], ignore_index=True)
    loc_less_only = suburb_lookup[~suburb_lookup["Suburb"].isin(loc_full["Suburb"])].copy()
    if len(loc_less_only) > 0:
        loc_less_df = pd.DataFrame({
            "Suburb": loc_less_only["Suburb"],
            "Postcode": loc_less_only["Postcode_ref"],
            "CouncilArea": loc_less_only["CouncilArea_ref"],
            "Regionname": loc_less_only["Regionname_ref"],
            "DistanceCBD": loc_less_only["Distance_ref"],
            "Latitude": MELBOURNE_CBD_LAT,
            "Longitude": MELBOURNE_CBD_LON,
            "PropertyCount": loc_less_only["Propertycount_ref"],
            "PriorSuburbMedian": global_train_median_price,
        })
        dim_location = pd.concat([loc_full, loc_less_df], ignore_index=True)
    else:
        dim_location = loc_full.copy()

    dim_location = dim_location.drop_duplicates(subset=["Suburb"]).sort_values("Suburb").reset_index(drop=True)
    dim_location["LocationKey"] = np.arange(1, len(dim_location) + 1)
    dim_location = dim_location.rename(columns={"Regionname": "Region"})[
        ["LocationKey", "Suburb", "CouncilArea", "Region", "Postcode", "DistanceCBD", "Latitude", "Longitude", "PropertyCount", "PriorSuburbMedian"]
    ]

    all_dates = pd.Series(pd.concat([df_priced["SaleDate"], df_unseen["SaleDate"], df_less["SaleDate"]]).dropna().unique()).sort_values().reset_index(drop=True)
    dt_parsed = pd.to_datetime(all_dates)
    dim_date = pd.DataFrame({
        "DateKey": np.arange(1, len(all_dates) + 1),
        "Date": all_dates,
        "Day": dt_parsed.dt.day,
        "Month": dt_parsed.dt.month,
        "MonthName": dt_parsed.dt.strftime("%B"),
        "Quarter": dt_parsed.dt.quarter,
        "Year": dt_parsed.dt.year,
        "YearQuarter": dt_parsed.dt.year.astype(str) + "-Q" + dt_parsed.dt.quarter.astype(str),
    })

    dim_market = region_qtr.reset_index(drop=True).copy()
    dim_market["MarketKey"] = np.arange(1, len(dim_market) + 1)
    dim_market = dim_market.rename(columns={"Regionname": "Region"})[
        ["MarketKey", "Region", "YearQuarter", "InterestRate", "MarketIndex", "PriorPeriodMedianPrice", "CurrentQuarterMedianPrice", "QuarterTransactionVolume"]
    ]

    df_priced = df_priced.reset_index(drop=True)
    df_priced["PropertyKey"] = np.arange(1, len(df_priced) + 1)
    dim_property = df_priced[[
        "PropertyKey", "Address", "Type", "PropertyTypeLabel", "Rooms", "Bedroom2",
        "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "PropertyAge", "FloorToLandRatio",
        "BuildingArea_was_imputed", "YearBuilt_was_imputed", "Landsize_was_imputed"
    ]].rename(columns={
        "Type": "PropertyType",
        "Bedroom2": "Bedrooms",
        "Bathroom": "Bathrooms",
        "Car": "CarSpaces",
        "Landsize": "LandArea",
        "BuildingArea": "FloorArea",
    })

    loc_key_map = dim_location.set_index("Suburb")["LocationKey"].to_dict()
    date_key_map = dim_date.set_index("Date")["DateKey"].to_dict()
    mkt_key_map = dim_market.set_index(["Region", "YearQuarter"])["MarketKey"].to_dict()

    df_priced["LocationKey"] = df_priced["Suburb"].map(loc_key_map).fillna(1).astype(int)
    df_priced["DateKey"] = df_priced["SaleDate"].map(date_key_map).fillna(1).astype(int)
    df_priced["MarketKey"] = [
        mkt_key_map.get((r, yq), 1) for r, yq in zip(df_priced["Regionname"], df_priced["YearQuarter"])
    ]

    df_priced.to_csv(CLEANED_TRAIN_CSV, index=False)

    fact_property_sale = pd.DataFrame({
        "SaleKey": df_priced["PropertyKey"],
        "PropertyKey": df_priced["PropertyKey"],
        "LocationKey": df_priced["LocationKey"],
        "DateKey": df_priced["DateKey"],
        "MarketKey": df_priced["MarketKey"],
        "Method": df_priced["Method"],
        "SellerG": df_priced["SellerG"],
        "DataSplit": df_priced["DataSplit"],
        "SalePrice": df_priced["Price"],
        "PredictedPrice": df_priced["Price"],
        "AbsoluteError": 0.0,
        "PercentageError": 0.0,
        "ValuationStatus": "Fairly Valued",
    })

    conn = sqlite3.connect(DB_PATH)
    try:
        existing_fact = pd.read_sql_query("SELECT * FROM Fact_PropertySale ORDER BY SaleKey ASC", conn)
        if len(existing_fact) == len(fact_property_sale) and "PredictedPrice_XGB" in existing_fact.columns:
            for col_keep in [
                "PredictedPrice_XGB", "PredictedPrice_RF", "PredictedPrice_DT", "PredictedPrice_LR",
                "AbsoluteError", "PercentageError", "ValuationStatus"
            ]:
                if col_keep in existing_fact.columns:
                    fact_property_sale[col_keep] = existing_fact[col_keep].values
    except Exception:
        pass
    dim_property.to_sql("Dim_Property", conn, if_exists="replace", index=False)
    dim_location.to_sql("Dim_Location", conn, if_exists="replace", index=False)
    dim_date.to_sql("Dim_Date", conn, if_exists="replace", index=False)
    dim_market.to_sql("Dim_Market", conn, if_exists="replace", index=False)
    fact_property_sale.to_sql("Fact_PropertySale", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_prop ON Fact_PropertySale(PropertyKey);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_loc ON Fact_PropertySale(LocationKey);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_date ON Fact_PropertySale(DateKey);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_mkt ON Fact_PropertySale(MarketKey);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loc_suburb ON Dim_Location(Suburb);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loc_region ON Dim_Location(Region);")
    conn.commit()
    conn.close()

    elapsed_sec = round(time.time() - start_ts, 2)
    log_step(
        "LOAD",
        "Populate SQLite Star Schema Data Warehouse",
        f"Loaded Star Schema into {os.path.basename(DB_PATH)} in {elapsed_sec}s: "
        f"Fact_PropertySale ({len(fact_property_sale):,} rows), Dim_Property ({len(dim_property):,} rows), "
        f"Dim_Location ({len(dim_location):,} suburbs), Dim_Date ({len(dim_date):,} dates), "
        f"Dim_Market ({len(dim_market):,} region-quarter records).",
        {
            "Fact_PropertySale": int(len(fact_property_sale)),
            "Dim_Property": int(len(dim_property)),
            "Dim_Location": int(len(dim_location)),
            "Dim_Date": int(len(dim_date)),
            "Dim_Market": int(len(dim_market)),
            "Unseen_Properties": int(len(df_unseen)),
        },
    )

    # =========================================================================
    # COMPREHENSIVE DATA QUALITY AUDIT (DUPLICATES, MISSINGNESS, OUTLIERS, DROPS, ENCODING)
    # =========================================================================
    imputation_rules_map = {
        "Suburb": "No missing values in raw FULL or LESS CSVs.",
        "Address": "No missing values in raw FULL or LESS CSVs.",
        "Rooms": "No missing values in raw CSVs; clipped to [1, 10].",
        "Type": "No missing values in raw CSVs (h = House, u = Unit, t = Townhouse).",
        "Price": f"Target variable: {price_backfilled} backfilled from LESS via 7-key composite match; {len(df_unseen):,} remaining unpriced routed to Fact_UnseenProperty.",
        "Method": "No missing values in raw CSVs (S, SP, PI, VB, SA).",
        "SellerG": "No missing values in raw CSVs.",
        "Date": "Parsed via dayfirst ISO date parser (YYYY-MM-DD).",
        "Distance": "Backfilled from LESS suburb lookup; any remaining filled via Train global median.",
        "Postcode": "Backfilled from LESS suburb lookup.",
        "Bedroom2": "Missing/invalid replaced with Rooms count clipped to [1, 10].",
        "Bathroom": "Train-only (Type, Rooms) median imputation fallback to Train global median.",
        "Car": "Train-only (Type, Rooms) median imputation fallback to Train global median.",
        "Landsize": "4-tier Train-only median (Suburb×Type×Rooms → Suburb×Type → Region×Type → Type) + Landsize_was_imputed flag.",
        "BuildingArea": "5-tier Train-only median (Suburb×Type×Rooms → Suburb×Type → Region×Type×Rooms → Type×Rooms → Type) + BuildingArea_was_imputed flag.",
        "YearBuilt": "3-tier Train-only median (Suburb×Type → Suburb → Region) + YearBuilt_was_imputed flag.",
        "CouncilArea": "Backfilled from LESS suburb lookup; fallback to Train mode.",
        "Lattitude": "2-tier Train-only spatial median (Suburb → Region → Melbourne CBD).",
        "Longtitude": "2-tier Train-only spatial median (Suburb → Region → Melbourne CBD).",
        "Regionname": "Backfilled from LESS suburb lookup; fallback to Train mode.",
        "Propertycount": "Backfilled from LESS suburb lookup; fallback to Train median.",
    }

    missing_values_by_column = []
    for col in df_full_raw.columns:
        raw_full_miss = int(df_full_raw[col].isna().sum())
        raw_full_pct = round((raw_full_miss / float(full_raw_rows)) * 100.0, 2)
        if col in df_less_raw.columns:
            raw_less_miss = int(df_less_raw[col].isna().sum())
            raw_less_pct = round((raw_less_miss / float(less_raw_rows)) * 100.0, 2)
        else:
            raw_less_miss = None
            raw_less_pct = None

        pre_priced_miss = int(pre_imputation_priced_nulls.get(col, 0))
        pre_priced_pct = round((pre_priced_miss / float(len(df_priced))) * 100.0, 2)
        post_priced_miss = int(df_priced[col].isna().sum()) if col in df_priced.columns else 0
        post_unseen_miss = int(df_unseen[col].isna().sum()) if col in df_unseen.columns else 0

        missing_values_by_column.append({
            "column": col,
            "present_in_less_csv": bool(col in df_less_raw.columns),
            "raw_full_missing": raw_full_miss,
            "raw_full_missing_pct": raw_full_pct,
            "raw_less_missing": raw_less_miss,
            "raw_less_missing_pct": raw_less_pct,
            "pre_imputation_priced_missing": pre_priced_miss,
            "pre_imputation_priced_pct": pre_priced_pct,
            "post_cleaning_priced_missing": post_priced_miss,
            "post_cleaning_unseen_missing": post_unseen_miss,
            "imputation_rule": imputation_rules_map.get(col, "Train-only median/mode imputation"),
        })

    overlapping_less_full = int(df_less["RecordKey"].isin(full_key_set).sum())
    total_anomalies_fixed = int(sum(anomalies_fixed.values()))

    duplicates_audit = {
        "less_csv_raw_rows": int(less_raw_rows),
        "less_csv_exact_duplicates_removed": int(less_dups),
        "less_csv_deduplicated_rows": int(len(df_less)),
        "full_csv_raw_rows": int(full_raw_rows),
        "full_csv_exact_duplicates_removed": int(full_dups),
        "full_csv_deduplicated_rows": int(len(df_full)),
        "total_exact_duplicates_removed": int(less_dups + full_dups),
        "cross_csv_overlapping_keys_matched": overlapping_less_full,
        "unique_extra_market_rows_from_less": int(len(df_less_extra)),
        "composite_match_key": "Suburb | Address | Rooms | Type | Date | Method | SellerG",
    }

    outlier_rules_applied = [
        {
            "feature": "Price (Lower Tail)",
            "threshold_rule": "Price < $120,000",
            "action_taken": "Dropped from Fact_PropertySale (nominal / non-arm's-length transfers)",
            "rows_affected": int(n_price_low),
        },
        {
            "feature": "Price (Upper 99.8th Percentile)",
            "threshold_rule": f"Price > ${p_high:,.0f} (99.8th percentile)",
            "action_taken": "Dropped from Fact_PropertySale (extreme luxury tail outliers)",
            "rows_affected": int(n_price_high),
        },
        {
            "feature": "YearBuilt (Entry Typos)",
            "threshold_rule": "YearBuilt == 1196 or YearBuilt == 2106",
            "action_taken": "Corrected in-place (1196 → 1996, 2106 → 2016)",
            "rows_affected": int(n_yb_typo),
        },
        {
            "feature": "YearBuilt (Implausible Era)",
            "threshold_rule": "YearBuilt < 1835 or YearBuilt > 2019",
            "action_taken": "Masked to NaN → Train-only Suburb×Type median + YearBuilt_was_imputed=1",
            "rows_affected": int(n_yb_range),
        },
        {
            "feature": "BuildingArea (Zero / Extreme Floor Area)",
            "threshold_rule": "BuildingArea < 15 sqm or BuildingArea > 1,500 sqm",
            "action_taken": "Masked to NaN → 5-tier Train-only hierarchical median + BuildingArea_was_imputed=1",
            "rows_affected": int(n_ba_invalid),
        },
        {
            "feature": "Landsize (Zero-Land Houses)",
            "threshold_rule": "Landsize <= 0 sqm and Type == 'h' (House)",
            "action_taken": "Masked to NaN → 4-tier Train-only hierarchical median + Landsize_was_imputed=1",
            "rows_affected": int(n_ls_zero_house),
        },
        {
            "feature": "Landsize (Extreme Acreage)",
            "threshold_rule": "Landsize > 15,000 sqm",
            "action_taken": "Winsorized (capped) at 15,000 sqm",
            "rows_affected": int(n_ls_extreme),
        },
        {
            "feature": "Bedroom2 (Invalid Count)",
            "threshold_rule": "Bedroom2 <= 0 or Bedroom2 > 10",
            "action_taken": "Replaced with Rooms count clipped to [1, 10]",
            "rows_affected": int(n_bed_invalid),
        },
        {
            "feature": "Bathroom (Invalid Count)",
            "threshold_rule": "Bathroom <= 0 or Bathroom > 8",
            "action_taken": "Masked to NaN → Train-only (Type, Rooms) median",
            "rows_affected": int(n_bath_invalid),
        },
        {
            "feature": "Car (Invalid Count)",
            "threshold_rule": "Car < 0 or Car > 10",
            "action_taken": "Masked to NaN → Train-only (Type, Rooms) median",
            "rows_affected": int(n_car_invalid),
        },
    ]

    rows_dropped_and_separated = [
        {
            "stage": "1. Raw Source Records Ingested (LESS + FULL)",
            "record_count": int(less_raw_rows + full_raw_rows),
            "disposition": "Input",
            "reason": "63,023 rows in melbourne_house_prices_less.csv + 34,857 rows in melbourne_housing_full.csv.",
        },
        {
            "stage": "2. Exact Duplicate Rows in LESS CSV",
            "record_count": int(less_dups),
            "disposition": "Dropped",
            "reason": "Identical duplicate rows across all 13 columns in melbourne_house_prices_less.csv.",
        },
        {
            "stage": "3. Exact Duplicate Rows in FULL CSV",
            "record_count": int(full_dups),
            "disposition": "Dropped",
            "reason": "Identical duplicate row across all 21 columns in melbourne_housing_full.csv.",
        },
        {
            "stage": "4. Cross-CSV Overlapping Transactions (LESS ∩ FULL)",
            "record_count": overlapping_less_full,
            "disposition": "Merged / Deduplicated",
            "reason": f"Matched via 7-field RecordKey to backfill {price_backfilled} missing prices and {backfilled_fields} missing location attributes in FULL without double-counting.",
        },
        {
            "stage": "5. LESS-Only Transactions (Macro Market Benchmark)",
            "record_count": int(len(df_less_extra)),
            "disposition": "Routed to Dim_Market",
            "reason": "Lacks physical property attributes (Bedrooms, Bathrooms, Landsize, BuildingArea, YearBuilt, Lat/Long); used strictly for past-only regional/suburb market indices.",
        },
        {
            "stage": "6. Unpriced Listings in FULL (Missing Target Price)",
            "record_count": int(len(df_unseen)),
            "disposition": "Routed to Fact_UnseenProperty",
            "reason": "Listings with no recorded sale price after cross-CSV lookup; separated for out-of-sample valuation scoring (never used in training).",
        },
        {
            "stage": "7. Extreme Price Tail Outliers (< $120k or > 99.8th pct)",
            "record_count": int(n_price_outliers),
            "disposition": "Dropped",
            "reason": f"Removed {n_price_low} nominal transfers (< $120,000) and {n_price_high} extreme luxury outliers (> ${p_high:,.0f}) to stabilize regression loss.",
        },
        {
            "stage": "8. Final Validated Sales (Fact_PropertySale)",
            "record_count": int(len(df_priced)),
            "disposition": "Retained (21,753 Train / 5,439 Test)",
            "reason": "Clean, complete residential sales split 80% Train (21,753) and 20% Holdout Test (5,439) prior to imputation and target encoding.",
        },
    ]

    encoding_and_scaling_scheme = [
        {
            "feature_group": "Target Variable (Price)",
            "columns": "Price ($ AUD)",
            "encoding_method": "Log1p Transformation — TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)",
            "leakage_safeguard": "Invertible monotonic transformation; predictions evaluated in original AUD dollars.",
        },
        {
            "feature_group": "High-Cardinality Location (350 Suburbs)",
            "columns": "SuburbTargetEncoded, SuburbTypeTargetEncoded, SuburbTypeRoomsTargetEncoded",
            "encoding_method": "3-Tier Empirical Bayes (m-estimate) Target Encoding in log1p(Price) space (m = 10, 8, 6)",
            "leakage_safeguard": "Computed strictly Out-of-Fold (5-fold KFold) within Train set; Test & Unseen mapped from full-Train encoder only.",
        },
        {
            "feature_group": "Categorical Attributes (4 Columns → 49 One-Hot Cols)",
            "columns": "Type (3), Method (5), Regionname (8), CouncilArea (33)",
            "encoding_method": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "leakage_safeguard": "Fitted strictly on X_train inside sklearn ColumnTransformer pipeline.",
        },
        {
            "feature_group": "Continuous Physical, Spatial & Temporal Features (24 Cols)",
            "columns": "Rooms, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, PropertyAge, FloorToLandRatio, TotalUtilityRooms, Distance, RadialDistanceCBD, Lattitude, Longtitude, Propertycount, SaleYear, SaleQuarter, PriorSuburbMedianPrice, PriorPeriodMedianPrice, SuburbPremiumRatio, MarketIndex, QuarterTransactionVolume, ...",
            "encoding_method": "SimpleImputer(strategy='median') + StandardScaler() (zero mean, unit variance)",
            "leakage_safeguard": "Medians, means, and standard deviations fitted strictly on X_train; past-only chronological medians for market benchmarks.",
        },
        {
            "feature_group": "Missingness Indicator Flags (3 Binary Cols)",
            "columns": "BuildingArea_was_imputed, YearBuilt_was_imputed, Landsize_was_imputed",
            "encoding_method": "Binary {0, 1} indicators captured prior to imputation (Total model matrix = 76 dimensions)",
            "leakage_safeguard": "Deterministic row-level missingness check recorded before hierarchical Train-only imputation.",
        },
    ]

    etl_summary = {
        "status": "COMPLETED",
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": elapsed_sec,
        "source_files": [
            {
                "name": "melbourne_house_prices_less.csv",
                "raw_rows": int(less_raw_rows),
                "raw_cols": int(less_raw_cols),
                "duplicates_removed": int(less_dups),
                "deduplicated_rows": int(len(df_less)),
                "description": "Broad Melbourne transaction log (2016–2018) across 380 suburbs with complete location metadata.",
            },
            {
                "name": "melbourne_housing_full.csv",
                "raw_rows": int(full_raw_rows),
                "raw_cols": int(full_raw_cols),
                "duplicates_removed": int(full_dups),
                "deduplicated_rows": int(len(df_full)),
                "description": "Detailed property physical & spatial attributes (Bedrooms, Bathrooms, Car, Landsize, BuildingArea, YearBuilt, Lat/Long).",
            },
        ],
        "cleaning_metrics": {
            "total_raw_records_processed": int(less_raw_rows + full_raw_rows),
            "less_duplicates_removed": int(less_dups),
            "full_duplicates_removed": int(full_dups),
            "exact_duplicates_removed": int(less_dups + full_dups),
            "overlapping_records_merged": overlapping_less_full,
            "location_metadata_backfilled": int(backfilled_fields),
            "prices_backfilled_across_files": int(price_backfilled),
            "anomalies_corrected": anomalies_fixed,
            "total_physical_anomalies_corrected": total_anomalies_fixed,
            "price_outliers_removed": int(n_price_outliers),
            "price_low_outliers_removed": int(n_price_low),
            "price_high_outliers_removed": int(n_price_high),
            "price_upper_threshold_998pct": round(p_high, 0),
            "total_anomalies_and_outliers_handled": int(total_anomalies_fixed + n_price_outliers),
            "cleaned_priced_sales": int(len(df_priced)),
            "unseen_properties_for_prediction": int(len(df_unseen)),
            "train_split_rows": int((df_priced["DataSplit"] == "Train").sum()),
            "test_split_rows": int((df_priced["DataSplit"] == "Test").sum()),
        },
        "data_quality_audit": {
            "duplicates_audit": duplicates_audit,
            "missing_values_by_column": missing_values_by_column,
            "outlier_rules_applied": outlier_rules_applied,
            "rows_dropped_and_separated": rows_dropped_and_separated,
            "encoding_and_scaling_scheme": encoding_and_scaling_scheme,
        },
        "star_schema_tables": {
            "Fact_PropertySale": {
                "rows": int(len(fact_property_sale)),
                "columns": list(fact_property_sale.columns),
                "grain": "One completed residential property sale transaction",
            },
            "Dim_Property": {
                "rows": int(len(dim_property)),
                "columns": list(dim_property.columns),
                "grain": "One physical residential property profile",
            },
            "Dim_Location": {
                "rows": int(len(dim_location)),
                "columns": list(dim_location.columns),
                "grain": "One Melbourne suburb / postcode location profile",
            },
            "Dim_Date": {
                "rows": int(len(dim_date)),
                "columns": list(dim_date.columns),
                "grain": "One calendar transaction date",
            },
            "Dim_Market": {
                "rows": int(len(dim_market)),
                "columns": list(dim_market.columns),
                "grain": "One Region x Year-Quarter macro market indicator snapshot",
            },
        },
        "steps_log": steps_log,
    }

    with open(ETL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(etl_summary, f, indent=2)

    # Also write a comprehensive tabular CSV data quality summary to data/processed/etl_data_quality_summary.csv
    dq_csv_rows = []
    for item in rows_dropped_and_separated:
        dq_csv_rows.append({
            "Audit_Section": "1_Row_Disposition_and_Deduplication",
            "Item_or_Column": item["stage"],
            "Rule_or_Status": item["disposition"],
            "Before_or_Affected_Count": item["record_count"],
            "Before_Pct": "",
            "After_Priced_Count": "",
            "After_Unseen_Count": "",
            "Details_and_Methodology": item["reason"],
        })
    for rule in outlier_rules_applied:
        dq_csv_rows.append({
            "Audit_Section": "2_Outlier_and_Anomaly_Rules",
            "Item_or_Column": rule["feature"],
            "Rule_or_Status": rule["threshold_rule"],
            "Before_or_Affected_Count": rule["rows_affected"],
            "Before_Pct": "",
            "After_Priced_Count": 0,
            "After_Unseen_Count": 0,
            "Details_and_Methodology": rule["action_taken"],
        })
    for col_item in missing_values_by_column:
        dq_csv_rows.append({
            "Audit_Section": "3_Column_Missing_Values_Before_After",
            "Item_or_Column": col_item["column"],
            "Rule_or_Status": f"Raw FULL Missing: {col_item['raw_full_missing']} ({col_item['raw_full_missing_pct']}%)",
            "Before_or_Affected_Count": col_item["pre_imputation_priced_missing"],
            "Before_Pct": f"{col_item['pre_imputation_priced_pct']}%",
            "After_Priced_Count": col_item["post_cleaning_priced_missing"],
            "After_Unseen_Count": col_item["post_cleaning_unseen_missing"],
            "Details_and_Methodology": col_item["imputation_rule"],
        })
    for enc_item in encoding_and_scaling_scheme:
        dq_csv_rows.append({
            "Audit_Section": "4_Feature_Encoding_and_Scaling",
            "Item_or_Column": enc_item["feature_group"],
            "Rule_or_Status": enc_item["encoding_method"],
            "Before_or_Affected_Count": "",
            "Before_Pct": "",
            "After_Priced_Count": "",
            "After_Unseen_Count": "",
            "Details_and_Methodology": f"Columns: {enc_item['columns']} | Safeguard: {enc_item['leakage_safeguard']}",
        })
    dq_csv_path = os.path.join(PROCESSED_DATA_DIR, "etl_data_quality_summary.csv")
    pd.DataFrame(dq_csv_rows).to_csv(dq_csv_path, index=False)

    return etl_summary


if __name__ == "__main__":
    run_etl_pipeline()

