"""
Flask Web Server & REST API Controller for the Adaptive Business Intelligence Platform.
"""

import os
import json
import sqlite3
import pandas as pd
from flask import Flask, Response, render_template, request, jsonify

from src.config import (
    PROCESSED_DATA_DIR,
    DB_PATH,
    UNSEEN_CSV,
    TEMPLATES_DIR,
    SALE_METHOD_LABELS,
)
from src.etl_pipeline import run_etl_pipeline
from src.ml_models import train_and_evaluate_models
from src.abi_engine import PropertyABIEngine

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config["TEMPLATES_AUTO_RELOAD"] = True
abi_engine = PropertyABIEngine()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_full_unseen_table():
    """Ensure Fact_UnseenProperty in SQLite contains all 7,607 scored unseen properties with SuburbTypeMedianPrice."""
    if not os.path.exists(DB_PATH) or not os.path.exists(UNSEEN_CSV):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cnt_row = conn.execute("SELECT COUNT(*) FROM Fact_UnseenProperty").fetchone()
        current_cnt = int(cnt_row[0]) if cnt_row else 0
        col_info = conn.execute("PRAGMA table_info(Fact_UnseenProperty)").fetchall()
        col_names = {r[1] for r in col_info}
        if current_cnt < 7000 or "SuburbTypeMedianPrice" not in col_names:
            df_unseen = pd.read_csv(UNSEEN_CSV)
            cols = [
                "Suburb", "Address", "Rooms", "Type", "PropertyTypeLabel", "Bathroom", "Car",
                "Landsize", "BuildingArea", "YearBuilt", "PropertyAge", "Regionname", "Distance",
                "SaleDate", "PredictedPrice_XGB", "PredictedPrice_RF", "SuburbTypeMedianPrice"
            ]
            avail_cols = [c for c in cols if c in df_unseen.columns]
            df_unseen[avail_cols].to_sql("Fact_UnseenProperty", conn, if_exists="replace", index=False)
            conn.commit()
        conn.close()
    except Exception:
        pass


ensure_full_unseen_table()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return real-time operational status of all 4 ML models, ETL pipeline, Star Schema DW, and AI Agent."""
    return jsonify(abi_engine.get_system_status())


@app.route("/api/options", methods=["GET"])
def api_options():
    """Return dropdown options for Suburbs, Regions, Councils, Property Types, Rooms, Sale Methods, and Models."""
    suburbs_list = []
    regions_map = {}
    councils_set = set()

    for sub_name, prof in sorted(abi_engine.suburb_profiles.items()):
        reg = prof.get("Regionname", "Southern Metropolitan")
        council = prof.get("CouncilArea", "Melbourne City Council")
        councils_set.add(council)
        sub_item = {
            "suburb": sub_name,
            "region": reg,
            "council_area": council,
            "postcode": int(prof.get("Postcode", 3000)),
            "distance_km": round(float(prof.get("Distance", 10.0)), 1),
            "median_price": round(float(prof.get("PriorSuburbMedianPrice", 900000.0)), 0),
            "type_medians": prof.get("TypeMedians", {}),
            "median_landsize": round(float(prof.get("MedianLandsize", 500.0)), 0),
            "median_building_area": round(float(prof.get("MedianBuildingArea", 140.0)), 0),
            "median_year_built": int(prof.get("MedianYearBuilt", 1975)),
            "sample_count": int(prof.get("SampleCount", 10)),
        }
        suburbs_list.append(sub_item)
        if reg not in regions_map:
            regions_map[reg] = {"region": reg, "suburb_count": 0, "total_sales": 0}
        regions_map[reg]["suburb_count"] += 1
        regions_map[reg]["total_sales"] += int(prof.get("SampleCount", 0))

    regions_list = sorted(regions_map.values(), key=lambda x: x["total_sales"], reverse=True)

    return jsonify({
        "regions": regions_list,
        "suburbs": suburbs_list,
        "councils": sorted(councils_set),
        "property_types": [
            {"code": "h", "label": "House / Cottage / Villa (h)"},
            {"code": "u", "label": "Unit / Duplex / Apartment (u)"},
            {"code": "t", "label": "Townhouse (t)"},
        ],
        "sale_methods": [
            {"code": k, "label": f"{v} ({k})"} for k, v in SALE_METHOD_LABELS.items()
        ],
        "models": [
            {"id": "xgboost", "name": "XGBoost Regression (Champion Model)"},
            {"id": "abi_adaptive", "name": "ABI Dynamic Adaptive Ensemble (XGB + RF + DT + LR)"},
            {"id": "random_forest", "name": "Random Forest Regression (Ensemble)"},
            {"id": "decision_tree", "name": "Decision Tree Regression (Rule-Based)"},
            {"id": "linear_regression", "name": "Linear Regression (Parametric Baseline)"},
        ],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Run live property price prediction across all 4 models + ABI Ensemble from dropdown selections."""
    payload = request.get_json(force=True, silent=True) or {}
    result = abi_engine.predict_property(payload)
    return jsonify(result)


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """PowerBI-style OLAP Star Schema query endpoint supporting cross-filtering slicers."""
    region = request.args.get("region", "ALL")
    suburb = request.args.get("suburb", "ALL")
    prop_type = request.args.get("property_type", "ALL")
    year = request.args.get("year", "ALL")
    rooms = request.args.get("rooms", "ALL")
    max_price = request.args.get("max_price", "ALL")
    val_status = request.args.get("valuation_status", "ALL")
    split = request.args.get("split", "TEST").upper()

    where_clauses = ["1=1"]
    params = []

    if split == "TEST":
        where_clauses.append("f.DataSplit = 'Test'")
    elif split == "TRAIN":
        where_clauses.append("f.DataSplit = 'Train'")

    if region and region != "ALL":
        where_clauses.append("l.Region = ?")
        params.append(region)
    if suburb and suburb != "ALL":
        where_clauses.append("l.Suburb = ?")
        params.append(suburb)
    if prop_type and prop_type != "ALL":
        where_clauses.append("p.PropertyType = ?")
        params.append(prop_type)
    if year and year != "ALL":
        where_clauses.append("d.Year = ?")
        params.append(int(year))
    if rooms and rooms != "ALL":
        if str(rooms).endswith("+"):
            where_clauses.append("p.Rooms >= ?")
            params.append(int(str(rooms).replace("+", "")))
        else:
            where_clauses.append("p.Rooms = ?")
            params.append(int(rooms))
    if max_price and max_price != "ALL":
        where_clauses.append("f.SalePrice <= ?")
        params.append(float(max_price))
    if val_status and val_status != "ALL":
        where_clauses.append("f.ValuationStatus = ?")
        params.append(val_status)

    where_sql = " AND ".join(where_clauses)
    conn = get_db_connection()

    kpi_sql = f"""
        SELECT
            COUNT(*) AS total_sales,
            AVG(f.SalePrice) AS avg_actual_price,
            AVG(f.PredictedPrice_XGB) AS avg_predicted_price,
            AVG(CASE WHEN f.DataSplit = 'Test' THEN f.AbsoluteError ELSE NULL END) AS test_mae_slice,
            AVG(f.AbsoluteError) AS mae_slice,
            AVG(ABS(f.PercentageError)) AS mape_slice,
            SUM(CASE WHEN f.ValuationStatus = 'Undervalued Opportunity' THEN 1 ELSE 0 END) AS undervalued_count,
            AVG(l.DistanceCBD) AS avg_distance_cbd,
            AVG(p.FloorArea) AS avg_floor_area,
            AVG(p.LandArea) AS avg_land_area
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
    """
    kpi_row = dict(conn.execute(kpi_sql, params).fetchone())

    group_col = "l.Suburb" if (region and region != "ALL") else "l.Region"
    regional_sql = f"""
        SELECT
            {group_col} AS label,
            COUNT(*) AS sales_count,
            ROUND(AVG(f.SalePrice), 0) AS avg_actual,
            ROUND(AVG(f.PredictedPrice_XGB), 0) AS avg_predicted,
            ROUND(AVG(f.AbsoluteError), 0) AS avg_error
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        GROUP BY {group_col}
        ORDER BY sales_count DESC
        LIMIT 12
    """
    regional_data = [dict(r) for r in conn.execute(regional_sql, params).fetchall()]

    trend_sql = f"""
        SELECT
            d.YearQuarter AS period,
            COUNT(*) AS sales_volume,
            ROUND(AVG(f.SalePrice), 0) AS avg_actual,
            ROUND(AVG(f.PredictedPrice_XGB), 0) AS avg_predicted,
            ROUND(AVG(m.InterestRate), 2) AS interest_rate,
            ROUND(AVG(m.MarketIndex), 1) AS market_index
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        JOIN Dim_Market m ON f.MarketKey = m.MarketKey
        WHERE {where_sql}
        GROUP BY d.YearQuarter
        ORDER BY d.YearQuarter ASC
    """
    trend_data = [dict(r) for r in conn.execute(trend_sql, params).fetchall()]

    type_sql = f"""
        SELECT
            p.PropertyType AS type_code,
            p.PropertyTypeLabel AS type_label,
            COUNT(*) AS sales_count,
            ROUND(AVG(f.SalePrice), 0) AS avg_actual,
            ROUND(AVG(f.PredictedPrice_XGB), 0) AS avg_predicted,
            ROUND(AVG(f.AbsoluteError), 0) AS avg_error
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        GROUP BY p.PropertyType, p.PropertyTypeLabel
        ORDER BY sales_count DESC
    """
    type_data = [dict(r) for r in conn.execute(type_sql, params).fetchall()]

    rooms_sql = f"""
        SELECT
            CASE WHEN p.Rooms >= 6 THEN '6+ Rooms' ELSE CAST(p.Rooms AS TEXT) || ' Rooms' END AS rooms_label,
            MIN(p.Rooms) AS sort_order,
            COUNT(*) AS sales_count,
            ROUND(AVG(f.SalePrice), 0) AS avg_actual,
            ROUND(AVG(f.PredictedPrice_XGB), 0) AS avg_predicted
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        GROUP BY rooms_label
        ORDER BY sort_order ASC
    """
    rooms_data = [dict(r) for r in conn.execute(rooms_sql, params).fetchall()]

    scatter_sql = f"""
        SELECT
            l.Suburb AS suburb,
            p.Address AS address,
            l.Region AS region,
            p.PropertyType AS type_code,
            p.PropertyTypeLabel AS type_label,
            p.Rooms AS rooms,
            f.DataSplit AS data_split,
            f.SalePrice AS actual_price,
            f.PredictedPrice_XGB AS predicted_price,
            f.AbsoluteError AS abs_error,
            f.ValuationStatus AS valuation_status
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        ORDER BY CASE WHEN f.DataSplit = 'Test' THEN 0 ELSE 1 END ASC, f.SaleKey % 19 ASC
        LIMIT 350
    """
    scatter_points = [dict(r) for r in conn.execute(scatter_sql, params).fetchall()]

    error_matrix_sql = f"""
        SELECT
            p.PropertyType AS type_code,
            p.PropertyTypeLabel AS type_label,
            CASE
                WHEN p.Rooms <= 1 THEN '1 Bed'
                WHEN p.Rooms = 2 THEN '2 Bed'
                WHEN p.Rooms = 3 THEN '3 Bed'
                WHEN p.Rooms = 4 THEN '4 Bed'
                ELSE '5+ Bed'
            END AS bedroom_group,
            MIN(p.Rooms) AS sort_order,
            COUNT(*) AS sales_count,
            ROUND(COALESCE(AVG(CASE WHEN f.DataSplit = 'Test' THEN f.AbsoluteError END), AVG(f.AbsoluteError)), 0) AS mean_abs_error,
            ROUND(AVG(ABS(f.PercentageError)), 1) AS mape_pct
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        GROUP BY p.PropertyType, p.PropertyTypeLabel, bedroom_group
        ORDER BY sort_order ASC, p.PropertyType ASC
    """
    error_by_type_rooms = [dict(r) for r in conn.execute(error_matrix_sql, params).fetchall()]

    geo_sql = f"""
        SELECT
            l.Suburb AS suburb,
            p.Address AS address,
            l.Region AS region,
            l.Latitude AS lat,
            l.Longitude AS lon,
            p.PropertyTypeLabel AS type_label,
            p.Rooms AS rooms,
            f.SalePrice AS actual_price,
            f.PredictedPrice_XGB AS predicted_price,
            f.ValuationStatus AS valuation_status
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql}
        ORDER BY f.SaleKey % 47 ASC
        LIMIT 350
    """
    geo_points = [dict(r) for r in conn.execute(geo_sql, params).fetchall()]

    undervalued_sql = f"""
        SELECT
            p.Address AS address,
            l.Suburb AS suburb,
            l.Region AS region,
            p.PropertyTypeLabel AS type_label,
            p.Rooms AS rooms,
            p.Bathrooms AS bathrooms,
            p.CarSpaces AS cars,
            ROUND(p.LandArea, 0) AS land_area,
            ROUND(p.FloorArea, 0) AS floor_area,
            d.Date AS sale_date,
            f.SalePrice AS actual_price,
            f.PredictedPrice_XGB AS predicted_price,
            ROUND(f.PredictedPrice_XGB - f.SalePrice, 0) AS equity_upside,
            ROUND(((f.PredictedPrice_XGB - f.SalePrice) / f.PredictedPrice_XGB) * 100.0, 1) AS discount_pct,
            f.ValuationStatus AS valuation_status
        FROM Fact_PropertySale f
        JOIN Dim_Property p ON f.PropertyKey = p.PropertyKey
        JOIN Dim_Location l ON f.LocationKey = l.LocationKey
        JOIN Dim_Date d ON f.DateKey = d.DateKey
        WHERE {where_sql} AND f.ValuationStatus = 'Undervalued Opportunity'
        ORDER BY (f.PredictedPrice_XGB - f.SalePrice) DESC
        LIMIT 12
    """
    undervalued_deals = [dict(r) for r in conn.execute(undervalued_sql, params).fetchall()]

    unseen_suburb_q = request.args.get("unseen_suburb", "").strip()
    unseen_sort = request.args.get("unseen_sort", "opportunity")

    unseen_where = ["1=1"]
    unseen_params = []
    if region and region != "ALL":
        unseen_where.append("u.Regionname = ?")
        unseen_params.append(region)
    if suburb and suburb != "ALL":
        unseen_where.append("u.Suburb = ?")
        unseen_params.append(suburb)
    if unseen_suburb_q:
        unseen_where.append("(LOWER(u.Suburb) LIKE ? OR LOWER(u.Address) LIKE ?)")
        unseen_params.extend([f"%{unseen_suburb_q.lower()}%", f"%{unseen_suburb_q.lower()}%"])
    if prop_type and prop_type != "ALL":
        unseen_where.append("u.Type = ?")
        unseen_params.append(prop_type)
    if rooms and rooms != "ALL":
        if str(rooms).endswith("+"):
            unseen_where.append("u.Rooms >= ?")
            unseen_params.append(int(str(rooms).replace("+", "")))
        else:
            unseen_where.append("u.Rooms = ?")
            unseen_params.append(int(rooms))
    if max_price and max_price != "ALL":
        unseen_where.append("u.PredictedPrice_XGB <= ?")
        unseen_params.append(float(max_price))

    unseen_where_sql = " AND ".join(unseen_where)
    unseen_cnt_row = conn.execute(
        f"SELECT COUNT(*) FROM Fact_UnseenProperty u WHERE {unseen_where_sql}",
        unseen_params,
    ).fetchone()
    unseen_filtered_count = int(unseen_cnt_row[0]) if unseen_cnt_row else 0

    type_med_expr = "COALESCE(u.SuburbTypeMedianPrice, l.PriorSuburbMedian, u.PredictedPrice_XGB)"
    order_clause = f"({type_med_expr} - u.PredictedPrice_XGB) DESC"
    outer_order = "(SuburbTypeMedianPrice - PredictedPrice_XGB) DESC"
    if unseen_sort == "price_asc":
        order_clause = "u.PredictedPrice_XGB ASC"
        outer_order = "PredictedPrice_XGB ASC"
    elif unseen_sort == "price_desc":
        order_clause = "u.PredictedPrice_XGB DESC"
        outer_order = "PredictedPrice_XGB DESC"
    elif unseen_sort == "suburb_asc":
        order_clause = "u.Suburb ASC, u.PredictedPrice_XGB ASC"
        outer_order = "Suburb ASC, PredictedPrice_XGB ASC"

    rank_filter = "1=1" if unseen_suburb_q else "suburb_rank = 1"
    clean_land_filter = f"{unseen_where_sql} AND (u.Type = 'u' OR u.Landsize BETWEEN 60 AND 2500)"

    unseen_sql = f"""
        WITH RankedUnseen AS (
            SELECT
                u.Suburb,
                u.Address,
                u.Regionname,
                u.Type,
                u.PropertyTypeLabel,
                u.Rooms,
                ROUND(u.Bathroom, 0) AS Bathroom,
                ROUND(u.Car, 0) AS Car,
                ROUND(u.Landsize, 0) AS Landsize,
                ROUND(u.BuildingArea, 0) AS BuildingArea,
                CAST(u.YearBuilt AS INTEGER) AS YearBuilt,
                ROUND(u.Distance, 1) AS Distance,
                u.SaleDate,
                ROUND(u.PredictedPrice_XGB, 0) AS PredictedPrice_XGB,
                ROUND(u.PredictedPrice_RF, 0) AS PredictedPrice_RF,
                ROUND({type_med_expr}, 0) AS SuburbTypeMedianPrice,
                ROUND(COALESCE(l.PriorSuburbMedian, u.PredictedPrice_XGB), 0) AS SuburbMedianPrice,
                ROW_NUMBER() OVER (
                    PARTITION BY u.Suburb
                    ORDER BY {order_clause}
                ) AS suburb_rank
            FROM Fact_UnseenProperty u
            LEFT JOIN Dim_Location l ON u.Suburb = l.Suburb
            WHERE {clean_land_filter}
        )
        SELECT
            Suburb, Address, Regionname, Type, PropertyTypeLabel, Rooms,
            Bathroom, Car, Landsize, BuildingArea, YearBuilt, Distance,
            SaleDate, PredictedPrice_XGB, PredictedPrice_RF, SuburbTypeMedianPrice, SuburbMedianPrice
        FROM RankedUnseen
        WHERE {rank_filter}
        ORDER BY {outer_order}
        LIMIT 15
    """
    unseen_rows = [dict(r) for r in conn.execute(unseen_sql, unseen_params).fetchall()]
    if len(unseen_rows) < 5:
        fallback_unseen_sql = f"""
            SELECT
                u.Suburb,
                u.Address,
                u.Regionname,
                u.Type,
                u.PropertyTypeLabel,
                u.Rooms,
                ROUND(u.Bathroom, 0) AS Bathroom,
                ROUND(u.Car, 0) AS Car,
                ROUND(u.Landsize, 0) AS Landsize,
                ROUND(u.BuildingArea, 0) AS BuildingArea,
                CAST(u.YearBuilt AS INTEGER) AS YearBuilt,
                ROUND(u.Distance, 1) AS Distance,
                u.SaleDate,
                ROUND(u.PredictedPrice_XGB, 0) AS PredictedPrice_XGB,
                ROUND(u.PredictedPrice_RF, 0) AS PredictedPrice_RF,
                ROUND({type_med_expr}, 0) AS SuburbTypeMedianPrice,
                ROUND(COALESCE(l.PriorSuburbMedian, u.PredictedPrice_XGB), 0) AS SuburbMedianPrice
            FROM Fact_UnseenProperty u
            LEFT JOIN Dim_Location l ON u.Suburb = l.Suburb
            WHERE {unseen_where_sql}
            ORDER BY {order_clause}
            LIMIT 15
        """
        unseen_rows = [dict(r) for r in conn.execute(fallback_unseen_sql, unseen_params).fetchall()]
    conn.close()

    resolved_mae = kpi_row.get("test_mae_slice")
    if resolved_mae is None:
        resolved_mae = kpi_row.get("mae_slice") or 0

    xgb_meta = abi_engine.ml_meta.get("models", {}).get("xgboost", {})
    xgb_test_metrics = xgb_meta.get("test_metrics", {})

    return jsonify({
        "active_split": split,
        "split_counts": {
            "Test": int(abi_engine.etl_meta.get("cleaning_metrics", {}).get("test_split_rows", 5439)),
            "Train": int(abi_engine.etl_meta.get("cleaning_metrics", {}).get("train_split_rows", 21753)),
            "All": int(abi_engine.etl_meta.get("cleaning_metrics", {}).get("cleaned_priced_sales", 27192)),
        },
        "holdout_metrics": xgb_test_metrics,
        "kpis": {
            "total_sales": int(kpi_row.get("total_sales") or 0),
            "avg_actual_price": round(float(kpi_row.get("avg_actual_price") or 0), 0),
            "avg_predicted_price": round(float(kpi_row.get("avg_predicted_price") or 0), 0),
            "mae_slice": round(float(resolved_mae), 0),
            "overall_mae_slice": round(float(kpi_row.get("mae_slice") or 0), 0),
            "mape_slice": round(float(kpi_row.get("mape_slice") or 0), 2),
            "undervalued_count": int(kpi_row.get("undervalued_count") or 0),
            "unseen_scored_total": int(abi_engine.etl_meta.get("cleaning_metrics", {}).get("unseen_properties_for_prediction", 7607)),
            "unseen_filtered_count": unseen_filtered_count,
            "avg_distance_cbd": round(float(kpi_row.get("avg_distance_cbd") or 0), 1),
            "avg_floor_area": round(float(kpi_row.get("avg_floor_area") or 0), 1),
            "avg_land_area": round(float(kpi_row.get("avg_land_area") or 0), 1),
        },
        "regional_chart": regional_data,
        "temporal_trend": trend_data,
        "property_type_chart": type_data,
        "rooms_chart": rooms_data,
        "scatter_points": scatter_points,
        "error_by_type_rooms": error_by_type_rooms,
        "geo_points": geo_points,
        "undervalued_deals": undervalued_deals,
        "unseen_predictions": unseen_rows,
    })


@app.route("/api/etl", methods=["GET"])
def api_etl():
    """Return ETL pipeline metadata and sample rows from the Star Schema tables."""
    table_name = request.args.get("table", "Fact_PropertySale")
    allowed_tables = {
        "Fact_PropertySale",
        "Dim_Property",
        "Dim_Location",
        "Dim_Date",
        "Dim_Market",
        "Fact_UnseenProperty",
    }
    if table_name not in allowed_tables:
        table_name = "Fact_PropertySale"

    conn = get_db_connection()
    sample_rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table_name} LIMIT 15").fetchall()]
    conn.close()

    return jsonify({
        "etl_metadata": abi_engine.etl_meta,
        "selected_table": table_name,
        "sample_rows": sample_rows,
    })


@app.route("/api/etl/quality-report", methods=["GET"])
def api_etl_quality_report():
    """Return downloadable ETL & Data Quality Summary in CSV or JSON format."""
    fmt = request.args.get("format", "json").lower().strip()
    abi_engine.reload_artifacts()
    etl_meta = abi_engine.etl_meta or {}

    if fmt == "csv":
        csv_path = os.path.join(PROCESSED_DATA_DIR, "etl_data_quality_summary.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_text = f.read()
        else:
            dq = etl_meta.get("data_quality_audit", {})
            rows = []
            for item in dq.get("rows_dropped_and_separated", []):
                rows.append({
                    "Audit_Section": "1_Row_Disposition_and_Deduplication",
                    "Item_or_Column": item.get("stage", ""),
                    "Rule_or_Status": item.get("disposition", ""),
                    "Before_or_Affected_Count": item.get("record_count", ""),
                    "Before_Pct": "",
                    "After_Priced_Count": "",
                    "After_Unseen_Count": "",
                    "Details_and_Methodology": item.get("reason", ""),
                })
            for rule in dq.get("outlier_rules_applied", []):
                rows.append({
                    "Audit_Section": "2_Outlier_and_Anomaly_Rules",
                    "Item_or_Column": rule.get("feature", ""),
                    "Rule_or_Status": rule.get("threshold_rule", ""),
                    "Before_or_Affected_Count": rule.get("rows_affected", ""),
                    "Before_Pct": "",
                    "After_Priced_Count": 0,
                    "After_Unseen_Count": 0,
                    "Details_and_Methodology": rule.get("action_taken", ""),
                })
            for col_item in dq.get("missing_values_by_column", []):
                rows.append({
                    "Audit_Section": "3_Column_Missing_Values_Before_After",
                    "Item_or_Column": col_item.get("column", ""),
                    "Rule_or_Status": f"Raw FULL Missing: {col_item.get('raw_full_missing', 0)} ({col_item.get('raw_full_missing_pct', 0)}%)",
                    "Before_or_Affected_Count": col_item.get("pre_imputation_priced_missing", 0),
                    "Before_Pct": f"{col_item.get('pre_imputation_priced_pct', 0)}%",
                    "After_Priced_Count": col_item.get("post_cleaning_priced_missing", 0),
                    "After_Unseen_Count": col_item.get("post_cleaning_unseen_missing", 0),
                    "Details_and_Methodology": col_item.get("imputation_rule", ""),
                })
            csv_text = pd.DataFrame(rows).to_csv(index=False)

        return Response(
            csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="melbourne_etl_data_quality_summary.csv"'
            },
        )

    json_payload = json.dumps(etl_meta, indent=2)
    return Response(
        json_payload,
        mimetype="application/json; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="melbourne_etl_data_quality_summary.json"'
        },
    )


@app.route("/download/report.docx", methods=["GET"])
def download_docx_report():
    """Serve downloadable Microsoft Word (.docx) technical report."""
    docx_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "COMP8811_Assignment2_Technical_Report.docx")
    if not os.path.exists(docx_file):
        from scripts.generate_docx_report import main as gen_main
        gen_main()
    with open(docx_file, "rb") as f:
        data = f.read()
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": 'attachment; filename="COMP8811_Assignment2_Technical_Report.docx"'
        },
    )


@app.route("/download/report.md", methods=["GET"])
def download_md_report():
    """Serve downloadable Markdown (.md) technical report."""
    md_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "technical_report.md")
    with open(md_file, "r", encoding="utf-8") as f:
        data = f.read()
    return Response(
        data,
        mimetype="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="COMP8811_Assignment2_Technical_Report.md"'
        },
    )


@app.route("/api/etl/run", methods=["POST"])
def api_etl_run():
    """Trigger a live execution of the ETL pipeline and reload data warehouse metadata."""
    summary = run_etl_pipeline()
    train_and_evaluate_models()
    abi_engine.reload_artifacts()
    return jsonify({
        "status": "SUCCESS",
        "message": "ETL Pipeline and 4 Regression ML Models executed and updated Star Schema successfully.",
        "etl_summary": summary,
    })


@app.route("/api/ml", methods=["GET"])
def api_ml():
    """Return complete ML evaluation metrics, confidence intervals, feature importances, and slice analysis."""
    return jsonify(abi_engine.ml_meta)


@app.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    """Conversational AI Agent endpoint accepting natural language property & ABI queries."""
    payload = request.get_json(force=True, silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "Please enter a requirement or question for the AI Agent."}), 400
    response = abi_engine.handle_chat(user_message)
    return jsonify(response)


@app.route("/api/abi/optimize", methods=["POST"])
def api_abi_optimize():
    """Adaptive Business Intelligence (ABI) Multi-Objective Portfolio & Acquisition Optimizer."""
    payload = request.get_json(force=True, silent=True) or {}
    result = abi_engine.run_abi_optimization(payload)
    return jsonify(result)


@app.route("/api/abi/adapt", methods=["POST"])
def api_abi_adapt():
    """Online Adaptive Learning Loop: submit new market transaction feedback to adapt suburb valuation."""
    payload = request.get_json(force=True, silent=True) or {}
    event = abi_engine.record_adaptive_feedback(payload)
    return jsonify(event)

