# Residential Property Price Prediction & BI Decision Support

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost%20Champion%20(81.9%25%20R%C2%B2)-emerald)
![Conformal](https://img.shields.io/badge/Uncertainty-Split%20Conformal%20(95.3%25%20Coverage)-blue)
![SQLite](https://img.shields.io/badge/Data%20Warehouse-Star%20Schema%20OLAP-purple?logo=sqlite)
![Flask](https://img.shields.io/badge/Backend-Flask%20REST%20API-lightgrey?logo=flask)
![Plotly](https://img.shields.io/badge/Frontend-Plotly.js%20BI%20Dashboard-amber)

An academic and enterprise-grade **Machine Learning, Dimensional Star Schema Data Warehouse, and Business Intelligence (BI) Decision Support System** built for the Melbourne residential housing market (COMP8811 Assignment 2, Bayleys Real Estate Case Study).

The platform combines an automated **dual-CSV leak-free ETL pipeline**, a **5-table relational SQLite Star Schema OLAP warehouse**, **4 tuned regression models** (*Linear Regression, Decision Tree, Random Forest, and XGBoost*), instance-level **TreeSHAP hedonic explainability**, **Split Conformal Prediction intervals**, and a **PowerBI-style executive analytical dashboard**.

---

## Key Platform Features

1. **Dual-Source Leak-Free ETL Pipeline (`src/etl_pipeline.py`)**
   - Ingests **97,880 raw records** from `melbourne_house_prices_less.csv` (63,023 rows) and `melbourne_housing_full.csv` (34,857 rows).
   - Reconciles 34,855 cross-file overlapping transactions using a 7-key composite identifier (`Suburb|Address|Rooms|Type|Date|Method|SellerG`), backfilling 11 missing location metadata fields and 2 missing prices.
   - Enforces **10 explicit outlier and anomaly detection rules** (504 physical anomalies normalized; 57 extreme price tail outliers dropped where $\text{Price} < \$120,000$ or $> \$5,043,024$).
   - Performs a deterministic 80/20 train/test split (`random_state=42`) **strictly before fitting any imputer, scaler, or encoder**.
   - Applies 5-tier hierarchical median imputation fitted exclusively on the 21,753 training rows, paired with explicit binary missingness indicator flags (`*_was_imputed`).
   - Computes **5-fold out-of-fold Empirical Bayes target encoding** (`SuburbTargetEncoded`, `SuburbTypeTargetEncoded`, `SuburbTypeRoomsTargetEncoded`) in log-space within training folds to eliminate target leakage.

2. **Dimensional Star Schema Data Warehouse (`data/processed/property_star_schema.db`)**
   - Relational SQLite OLAP warehouse with indexed foreign keys:
     - `Fact_PropertySale`: **27,192** validated sales with actual sale prices, 4 model predictions, errors, and evaluation split flags (`21,753` Train / `5,439` Holdout Test).
     - `Fact_UnseenProperty`: **7,607** unpriced listings scored out-of-sample and benchmarked against same-suburb and same-dwelling-type medians (`SuburbTypeMedianPrice`).
     - `Dim_Property`: **27,192** structural property profiles with physical dimensions, engineered ratios (`PropertyAge`, `FloorToLandRatio`), and imputation flags.
     - `Dim_Location`: **381** Melbourne suburb profiles with council area, region, postcodes, coordinates, and CBD radial distance.
     - `Dim_Date`: **112** discrete calendar settlement dates from 2016 through 2018.
     - `Dim_Market`: **96** regional quarterly macroeconomic snapshots with chronological past-only median prices, transaction volumes, and RBA cash rates.

3. **Multi-Model Regression Benchmark & Conformal Uncertainty (`src/ml_models.py`)**
   - All models trained with `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` to eliminate price skewness.
   - Hyperparameters tuned with `RandomizedSearchCV` and 5-fold cross-validation on the 21,753 training set.
   - Evaluated out-of-sample on the 5,439 Holdout Test partition ($R^2$, MAE, RMSE, MAPE) with 95% bootstrap confidence intervals.
   - Uncertainty quantified via **Split Conformal Prediction** stratified by dwelling type (`h`, `t`, `u`), achieving **95.26% empirical coverage** on unseen test data.

---

## Official Model Performance Benchmark

*Evaluated on the unseen 20% Holdout Test Set ($n = 5,439$) in original AUD dollar scale (`random_state=42`).*

| Regression Model | Role | 5-Fold CV $R^2$ (`mean ± std`) | Holdout Test $R^2$ `[95% CI]` | 5-Fold CV MAE (`mean ± std`) | Holdout Test MAE `[95% CI]` | Holdout Test RMSE | Holdout Test MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regression** | **Champion Model** | **82.39% ± 0.51%** | **81.88%** `[79.98%, 83.85%]` | **\$149,786 ± \$3,971** | **\$144,337** `[$139,425, $148,778]` | **\$249,455** | **13.37%** |
| **Random Forest Regression** | Bagging Ensemble | 80.26% ± 0.35% | 78.40% `[76.17%, 80.94%]` | \$159,637 ± \$3,384 | \$158,062 `[$152,355, $162,445]` | \$272,413 | 14.60% |
| **Linear Regression** | Log-Linear Baseline | 75.18% ± 1.07% | 74.19% `[72.52%, 76.39%]` | \$182,405 ± \$3,901 | \$181,931 `[$175,984, $186,509]` | \$297,782 | 17.04% |
| **Decision Tree Regression** | Rule-Based Trees | 74.59% ± 0.58% | 72.58% `[70.21%, 75.43%]` | \$186,530 ± \$3,055 | \$185,924 `[$179,675, $191,835]` | \$306,919 | 17.35% |
| **Naive Mean Baseline** | Unconditional Mean | — | -0.01% | — | \$437,766 | \$586,133 | 52.34% |

---

## Repository Structure

```text
├── README.md                                  # Executive platform overview & reproducibility guide
├── requirements.txt                           # Pinned Python package dependencies
├── run.py                                     # Live web application entry point (http://127.0.0.1:5000)
├── src/                                       # Core application source code
│   ├── config.py                              # Central configuration, file paths & feature schemas
│   ├── etl_pipeline.py                        # Leak-free dual-CSV ETL & Star Schema builder
│   ├── ml_models.py                           # 5-fold CV, RandomizedSearchCV tuning & conformal calibration
│   ├── abi_engine.py                          # TreeSHAP decomposition, conformal intervals & prediction service
│   └── web_app.py                             # Flask REST API endpoints & Star Schema OLAP queries
├── web/
│   └── templates/
│       └── dashboard.html                     # Responsive 3-tab UI with Plotly.js charts & CSV/JSON exports
├── docs/                                      # COMP8811 Assignment 2 Report & Demo Support Deliverables
│   ├── architecture_diagram.mmd               # Mermaid system architecture (Sources → ETL → DW → ML → UI)
│   ├── flowchart.mmd                          # Mermaid program execution & user interaction flowcharts
│   ├── ui_screenshots_checklist.md            # Evidence screenshot checklist with ready-to-paste captions
│   ├── results_tables.md                      # Complete empirical tables (ETL, outliers, CV, residual analysis)
│   ├── limitations_and_future_work.md         # Critical evaluation of 2016-2018 data, missingness & luxury tail
│   └── demo_script.md                         # 5-7 minute walkthrough presentation script & marker Q&A defense
├── data/
│   ├── raw/                                   # Raw input datasets
│   │   ├── melbourne_house_prices_less.csv    # 63,023 broad transaction records
│   │   └── melbourne_housing_full.csv         # 34,857 detailed physical property records
│   └── processed/                             # Star Schema Data Warehouse & audit reports
│       ├── property_star_schema.db            # SQLite Star Schema warehouse (5 tables + indexes)
│       ├── etl_pipeline_report.json           # Comprehensive JSON ETL audit metrics
│       ├── etl_data_quality_summary.csv       # Tabular CSV data quality report
│       └── ml_evaluation_report.json          # Multi-model CV, holdout metrics & slice diagnostics
├── models/                                    # Serialized scikit-learn & XGBoost pipelines (.joblib)
│   ├── feature_preprocessor.joblib            # Fitted ColumnTransformer pipeline (76 dimensions)
│   ├── suburb_market_profiles.joblib          # Suburb benchmarks & target encoding lookups
│   ├── linear_regression.joblib               # Trained Linear Regression model
│   ├── decision_tree_regressor.joblib         # Tuned Decision Tree model
│   ├── random_forest_regressor.joblib         # Tuned Random Forest model
│   └── xgboost_regressor.joblib               # Tuned Champion XGBoost model
└── tests/
    └── test_platform_api.py                   # Automated API, Star Schema & ML validation suite
```

---

## Quickstart & Reproducibility Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated API & System Verification
To verify all REST endpoints, Star Schema tables, out-of-sample metrics, TreeSHAP drivers, and data quality downloads:
```bash
python tests/test_platform_api.py
```

### 3. Launch the Interactive Web Application
```bash
python run.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### 4. Optional: Re-run ETL & Model Retraining
All pre-trained models and Star Schema tables are included in the repository. To re-run the full pipeline from scratch:
```bash
# Rebuild the Star Schema Data Warehouse & Data Quality Audit (takes ~7s):
python -m src.etl_pipeline

# Retrain all 4 models with 5-fold cross-validation and hyperparameter search (takes ~6-8m):
python -m src.ml_models
```

---

## Reproducibility & Integrity Notes
- **Random State:** All splits (`KFold`, `train_test_split`, `RandomizedSearchCV`) use a fixed seed of `42`.
- **Target Inversion:** Errors and predictions are reported in original Australian dollars ($ AUD) after $\exp(\hat{z}) - 1$ target inversion.
- **Conformal Calibration:** 95% prediction intervals are strictly calibrated on 21,753 held-out out-of-fold calibration residuals stratified across dwelling types (`h`, `t`, `u`).
