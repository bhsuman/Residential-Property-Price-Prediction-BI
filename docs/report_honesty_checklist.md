# COMP8811 Assignment 2 — Final Report Honesty & Verification Checklist

This checklist provides a line-by-line audit verifying that every empirical claim, architectural diagram, and modeling procedure presented in the COMP8811 Assignment 2 report corresponds to the codebase, database, and verified test results.

---

## 1. Academic Honesty & Anti-Leakage Verification

| Item | Status | Verification Evidence in Codebase / Database |
| :--- | :---: | :--- |
| **No In-Sample Contamination** | Verified | In `src/etl_pipeline.py`, the 80/20 train/test split (`random_state=42`) is executed at line 220, **before** fitting any missing-value imputer, target encoder, or scaler. |
| **Train-Only Imputation** | Verified | The 5-tier hierarchical median imputer (`src/etl_pipeline.py` lines 327–417) is calculated strictly on `train_ref` (`21,753` rows). No test statistics leak into training or vice-versa. |
| **Chronological Past-Only Medians** | Verified | In `src/etl_pipeline.py` lines 233–325, `PriorPeriodMedianPrice` and `PriorSuburbMedianPrice` are computed using expanding historical medians shifted by 1 date/quarter (`shift(1)`), eliminating same-day or future transaction leakage. |
| **Fold-Safe Target Encoding** | Verified | Suburb target encoding (`src/etl_pipeline.py` lines 495–620) is computed strictly using **5-fold out-of-fold cross-validation** within the training partition. Test and Unseen rows are mapped strictly from the fitted full-train encoder. |
| **Dashboard Default Evaluation** | Verified | The BI Dashboard (`src/web_app.py` line 147 and `web/templates/dashboard.html` line 526) **defaults strictly to the 5,439 Holdout Test partition** with an explicit `Holdout Test Only (5,439 Sales · Zero In-Sample Leakage)` status badge. |
| **Honest Interest Rate Framing** | Verified | The interest rate input on Tab 1 is explicitly labeled as `Interest Rate Scenario (Display Only · Not a Model Feature)` in `web/templates/dashboard.html` line 156, avoiding artificial post-hoc multipliers. |
| **Same-Type Unseen Benchmarking** | Verified | Unseen listings in `Fact_UnseenProperty` are compared against `SuburbTypeMedianPrice` (the training median for the **same suburb and dwelling type**) rather than the all-dwelling suburb median, using honest labels (`Below Typical for Type`, `Typical Range for Type`, `Above Typical for Type`). |

---

## 2. Empirical Model Metrics Verification

*All metrics are verified from `data/processed/ml_evaluation_report.json` and evaluated on the 5,439 Holdout Test set in original AUD dollars.*

| Regression Algorithm | 5-Fold CV $R^2$ (`mean ± std`) | Holdout Test $R^2$ | 5-Fold CV MAE (`mean ± std`) | Holdout Test MAE | Holdout Test RMSE | Holdout Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regression (Champion)** | **82.39% ± 0.51%** | **81.88%** | **\$149,786 ± \$3,971** | **\$144,337** | **\$249,455** | **13.37%** |
| **Random Forest Regression** | 80.26% ± 0.35% | 78.40% | \$159,637 ± \$3,384 | \$158,062 | \$272,413 | 14.60% |
| **Linear Regression (Log-Linear)** | 75.18% ± 1.07% | 74.19% | \$182,405 ± \$3,901 | \$181,931 | \$297,782 | 17.04% |
| **Decision Tree Regression** | 74.59% ± 0.58% | 72.58% | \$186,530 ± \$3,055 | \$185,924 | \$306,919 | 17.35% |
| **Naive Mean Baseline** | — | -0.01% | — | \$437,766 | \$586,133 | 52.34% |

- **Hyperparameters Tuned**: All 3 non-linear models were tuned via `RandomizedSearchCV` on the 21,753 training set (`n_iter=15`, 5-fold CV).
- **Target Invertibility**: All models use `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
- **Conformal Interval Calibration**: Nonconformity scores $s_i = |\log(1+y_i) - \log(1+\hat{y}_i^{\text{OOF}})|$ calibrated on 21,753 OOF training log-residuals achieve **95.26% empirical coverage** on the 5,439 Holdout Test set.

---

## 3. Data Cleaning & Star Schema Accounting

| Metric / Object | Reported in Text | Verified Database Count | Verification Query / File Reference |
| :--- | :---: | :---: | :--- |
| **Raw Records Ingested** | 97,880 | 97,880 | `63,023` in `melbourne_house_prices_less.csv` + `34,857` in `melbourne_housing_full.csv`. |
| **Exact Duplicates Removed** | 3 | 3 | `2` in `LESS` + `1` in `FULL` (`src/etl_pipeline.py` line 86). |
| **Cross-CSV Overlaps Merged** | 34,855 | 34,855 | `RecordKey` match between `LESS` and `FULL` (`src/etl_pipeline.py` line 139). |
| **Physical Anomalies Handled** | 504 | 504 | `YearBuilt` typos/range: 5; `BuildingArea`: 167; `Landsize`: 258; `Room/Bath/Car`: 74. |
| **Price Outliers Dropped** | 57 | 57 | `2` below $\$120,000$ + `55` above 99.8th percentile ($\$5,043,024$). |
| **Validated Sales (`Fact_PropertySale`)** | 27,192 | 27,192 | `SELECT COUNT(*) FROM Fact_PropertySale` in `property_star_schema.db`. |
| **Training Partition (80%)** | 21,753 | 21,753 | `SELECT COUNT(*) FROM Fact_PropertySale WHERE DataSplit='Train'`. |
| **Holdout Test Partition (20%)** | 5,439 | 5,439 | `SELECT COUNT(*) FROM Fact_PropertySale WHERE DataSplit='Test'`. |
| **Unpriced Listings (`Fact_UnseenProperty`)** | 7,607 | 7,607 | `SELECT COUNT(*) FROM Fact_UnseenProperty` in `property_star_schema.db`. |
| **Dim_Property Records** | 27,192 | 27,192 | `SELECT COUNT(*) FROM Dim_Property`. |
| **Dim_Location Suburbs** | 381 | 381 | `SELECT COUNT(*) FROM Dim_Location`. |
| **Dim_Date Settlement Dates** | 112 | 112 | `SELECT COUNT(*) FROM Dim_Date`. |
| **Dim_Market Region-Quarters** | 96 | 96 | `SELECT COUNT(*) FROM Dim_Market`. |

---

## 4. UI & Interactive Features Checklist

| UI Component | Location | Implementation & Verification Status |
| :--- | :--- | :--- |
| **Header Title & Status** | Global Header | Main title: *"Residential Property Price Prediction & BI Decision Support"*; subtitle: *"COMP8811 Assignment 2 — Predictive Analytics & Star Schema BI System (Bayleys Case Study \| Melbourne Housing Dataset)"*; Status pill: `4/4 Models Active`. |
| **Chatbot Buttons Removed** | Global Header & Body | Both the top-right and floating bottom-right concierge buttons/drawers were permanently removed to align strictly with the regression scope. |
| **Tab 1: Sample Profiles** | Tab 1 Top | 3 quick-fill buttons: `Richmond 3 Bed House`, `South Yarra 2 Bed Unit`, and `Kew 4 Bed Residence`. |
| **Tab 1: Hedonic Inputs** | Tab 1 Left Form | 9 clean inputs matching `Dim_Property` and `Dim_Location` (Region, Suburb, Type, Beds, Baths, Cars, Land Area, Floor Area, Year Built, plus Scenario Interest Rate). Out-of-scope budget ceiling and priority weights removed. |
| **Tab 1: Valuation Report** | Tab 1 Right Panel | Main price label: *"PREDICTED PROPERTY PRICE (ŷ — XGBoost Champion)"*; Split Conformal 95% interval with interactive tooltip; 4 academic BI metrics. |
| **Tab 1: Dynamic TreeSHAP** | Tab 1 Bottom-Left | Real per-prediction TreeSHAP contributions (`pred_contribs=True`) summing from baseline to $\hat{y}$, with local/global toggle. |
| **Tab 1: Cross-Model Comparison** | Tab 1 Bottom-Right | Side-by-side bar chart showing XGBoost, Random Forest, Decision Tree, and Linear Regression predictions with fully visible dollar labels. |
| **Tab 1: Comparable Settled Sales** | Tab 1 Bottom | 8 comparable property cards scored out-of-sample with `Holdout Test` and `5-Fold OOF` badges. |
| **Tab 2: BI Dashboard** | Tab 2 Top | Default: `Holdout Test Only (20% · 5,439)` with active split selector, 4 out-of-sample KPI cards, and 2×2 Plotly analytical grid. |
| **Tab 2: Unseen Listings Table** | Tab 2 Bottom | 7,607 listings benchmarked against `SuburbTypeMedianPrice` with honest relative badges and sorting options. |
| **Tab 3: Model Benchmark Cards** | Tab 3 Section 1 | 4 cards showing Holdout $R^2$, 5-Fold CV $R^2$ ($\text{mean} \pm \text{SD}$), Holdout MAE, Holdout RMSE, Holdout MAPE, and 5-Fold CV MAE ($\pm \text{SD}$). |
| **Tab 3: Residual Analysis** | Tab 3 Section 2 | 3 diagnostic charts: Residuals vs. Predicted scatter, Error by Price Band, and Error by Region/Type, plus the empirical finding callout. |
| **Tab 3: ETL Audit & Exports** | Tab 3 Section 3 | 4-panel data quality audit (duplicates/row disposition, outlier thresholds, missingness before/after, feature encoding), 1-click CSV/JSON download buttons, and Star Schema table inspector. |

---

## 5. Report Support Deliverables Map

The following 6 support documents have been authored and placed in the [`docs/`](file:///d:/Data%20Analytics/Property%20Analysis/docs/) directory:

1. [`docs/architecture_diagram.mmd`](file:///d:/Data%20Analytics/Property%20Analysis/docs/architecture_diagram.mmd) — System architecture diagram in Mermaid format.
2. [`docs/flowchart.mmd`](file:///d:/Data%20Analytics/Property%20Analysis/docs/flowchart.mmd) — Program execution and user interaction flowcharts in Mermaid format.
3. [`docs/ui_screenshots_checklist.md`](file:///d:/Data%20Analytics/Property%20Analysis/docs/ui_screenshots_checklist.md) — 7 exact UI screenshot captures with ready-to-paste report captions.
4. [`docs/results_tables.md`](file:///d:/Data%20Analytics/Property%20Analysis/docs/results_tables.md) — 9 clean Markdown tables covering dataset summaries, outliers, missingness, model comparisons, CV benchmarks, conformal intervals, feature importances, and residual slices.
5. [`docs/limitations_and_future_work.md`](file:///d:/Data%20Analytics/Property%20Analysis/docs/limitations_and_future_work.md) — Critical evaluation of the 2016–2018 market snapshot, missing physical attributes, unobserved hedonic drivers, luxury tail heteroscedasticity, and future engineering roadmap.
6. [`docs/demo_script.md`](file:///d:/Data%20Analytics/Property%20Analysis/docs/demo_script.md) — 5 to 7 minute live walkthrough presentation script and marker Q&A defense guide.

