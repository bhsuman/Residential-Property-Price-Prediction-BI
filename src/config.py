"""
Centralized configuration, file paths, domain mappings, and spatial helpers.
"""

import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "web", "templates")

LESS_CSV_PATH = os.path.join(RAW_DATA_DIR, "melbourne_house_prices_less.csv")
FULL_CSV_PATH = os.path.join(RAW_DATA_DIR, "melbourne_housing_full.csv")

DB_PATH = os.path.join(PROCESSED_DATA_DIR, "property_star_schema.db")
CLEANED_TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, "cleaned_property_sales.csv")
UNSEEN_CSV = os.path.join(PROCESSED_DATA_DIR, "unseen_properties_scored.csv")
ETL_META_PATH = os.path.join(PROCESSED_DATA_DIR, "etl_pipeline_report.json")
ML_META_PATH = os.path.join(PROCESSED_DATA_DIR, "ml_evaluation_report.json")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "feature_preprocessor.joblib")
SUBURB_PROFILES_PATH = os.path.join(MODELS_DIR, "suburb_market_profiles.joblib")

MODEL_ARTIFACT_FILES = {
    "linear_regression": os.path.join(MODELS_DIR, "linear_regression.joblib"),
    "decision_tree": os.path.join(MODELS_DIR, "decision_tree_regressor.joblib"),
    "random_forest": os.path.join(MODELS_DIR, "random_forest_regressor.joblib"),
    "xgboost": os.path.join(MODELS_DIR, "xgboost_regressor.joblib"),
}

# Historical Reserve Bank of Australia (RBA) Cash Rate (%) by Year-Quarter
RBA_INTEREST_RATES = {
    "2016-Q1": 2.00,
    "2016-Q2": 1.75,
    "2016-Q3": 1.50,
    "2016-Q4": 1.50,
    "2017-Q1": 1.50,
    "2017-Q2": 1.50,
    "2017-Q3": 1.50,
    "2017-Q4": 1.50,
    "2018-Q1": 1.50,
    "2018-Q2": 1.50,
    "2018-Q3": 1.50,
    "2018-Q4": 1.50,
}

PROPERTY_TYPE_LABELS = {
    "h": "House / Cottage / Villa",
    "u": "Unit / Duplex / Apartment",
    "t": "Townhouse",
}

SALE_METHOD_LABELS = {
    "S": "Property Sold",
    "SP": "Property Sold Prior",
    "PI": "Property Passed In",
    "VB": "Vendor Bid",
    "SN": "Sold Not Disclosed",
    "PN": "Sold Prior Not Disclosed",
    "SA": "Sold After Auction",
    "W": "Withdrawn Prior to Auction",
    "SS": "Sold After Auction Price Not Disclosed",
}

MELBOURNE_CBD_LAT = -37.8136
MELBOURNE_CBD_LON = 144.9631

NUMERIC_FEATURES = [
    "Rooms",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "PropertyAge",
    "FloorToLandRatio",
    "TotalUtilityRooms",
    "Distance",
    "RadialDistanceCBD",
    "Lattitude",
    "Longtitude",
    "Propertycount",
    "SaleYear",
    "SaleQuarter",
    "PriorPeriodMedianPrice",
    "PriorSuburbMedianPrice",
    "SuburbTargetEncoded",
    "SuburbTypeTargetEncoded",
    "SuburbTypeRoomsTargetEncoded",
    "SuburbPremiumRatio",
    "MarketIndex",
    "BuildingArea_was_imputed",
    "YearBuilt_was_imputed",
    "Landsize_was_imputed",
]

CATEGORICAL_FEATURES = [
    "Type",
    "Regionname",
    "Method",
    "CouncilArea",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine_km(lat1, lon1, lat2=MELBOURNE_CBD_LAT, lon2=MELBOURNE_CBD_LON):
    """Calculate great-circle radial distance in km from Melbourne CBD."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

