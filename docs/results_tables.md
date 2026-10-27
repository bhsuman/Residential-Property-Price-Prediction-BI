# COMP8811 Assignment 2 — Empirical Results & Audit Tables

All tables below contain exact, reproducible metrics generated from `data/processed/etl_pipeline_report.json` and `data/processed/ml_evaluation_report.json` (`random_state=42`). You can copy and paste any table directly into your COMP8811 Assignment 2 report.

---

## Table 1: Dual-Source Dataset & Star Schema Data Warehouse Summary

| Stage / Table | Record Count | Columns | Grain / Disposition |
| :--- | :---: | :---: | :--- |
| **Source 1: `melbourne_house_prices_less.csv`** | 63,023 | 13 | Raw Melbourne transaction log (2016–2018) across 380 suburbs (`2` exact duplicates removed → `63,021`). |
| **Source 2: `melbourne_housing_full.csv`** | 34,857 | 21 | Raw physical & spatial property dataset (`1` exact duplicate removed → `34,856`). |
| **Total Raw Records Ingested** | **97,880** | 13–21 | Combined dual-CSV ingestion prior to deduplication and cross-CSV reconciliation. |
| **Cross-CSV Overlapping Transactions (`LESS ∩ FULL`)** | 34,855 | 21 | Matched via 7-key composite key (`Suburb\|Address\|Rooms\|Type\|Date\|Method\|SellerG`); backfilled `2` missing prices and `11` missing location attributes in `FULL`. |
| **`LESS`-Only Market Transactions** | 28,166 | 13 | Lacks physical property attributes (`Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lat/Long`); routed strictly to `Dim_Market` for past-only regional/suburb price indices. |
| **Extreme Price Tail Outliers Dropped** | 57 | 21 | `2` nominal transfers (`Price < $120,000`) + `55` extreme luxury tail sales (`Price > $5,043,024` [99.8th percentile]). |
| **`Fact_PropertySale` (Validated Settled Sales)** | **27,192** | 16 | `21,753` Training rows (`80%`, scored via 5-Fold OOF) + `5,439` Holdout Test rows (`20%`, scored out-of-sample). |
| **`Fact_UnseenProperty` (Unpriced Listings Scored)** | **7,607** | 17 | Unpriced listings from `FULL` scored out-of-sample by XGBoost & Random Forest and compared against `SuburbTypeMedianPrice`. |
| **`Dim_Property`** | 27,192 | 16 | Physical property attributes, engineered ratios (`PropertyAge`, `FloorToLandRatio`), and `*_was_imputed` flags. |
| **`Dim_Location`** | 381 | 10 | Suburb, CouncilArea, Region, Postcode, DistanceCBD, Latitude, Longitude, PropertyCount, PriorSuburbMedian. |
| **`Dim_Date`** | 112 | 8 | Calendar settlement dates (`2016-01-28` to `2018-10-03`) with Day, Month, Quarter, Year, YearQuarter. |
| **`Dim_Market`** | 96 | 8 | Region × Year-Quarter macro snapshots (`PriorPeriodMedianPrice`, `MarketIndex`, RBA `InterestRate`, `QuarterTransactionVolume`). |

---

## Table 2: Outlier & Anomaly Detection Rules (Explicit Thresholds & Real Counts)

| Feature | Explicit Threshold Rule | Rows Affected | Disposition & Action Taken |
| :--- | :--- | :---: | :--- |
| **`Price` (Lower Tail)** | `Price < $120,000` | **2** | **Dropped** from `Fact_PropertySale` (non-arm's-length / nominal transfers). |
| **`Price` (Upper 99.8th Percentile)** | `Price > $5,043,024` (99.8th pct) | **55** | **Dropped** from `Fact_PropertySale` (extreme luxury tail outliers). |
| **`YearBuilt` (Known Data Entry Typos)** | `YearBuilt == 1196` or `YearBuilt == 2106` | **2** | **Corrected in-place** (`1196 → 1996`, `2106 → 2016`). |
| **`YearBuilt` (Implausible Historical/Future Era)** | `YearBuilt < 1835` or `YearBuilt > 2019` | **3** | **Masked to `NaN`** → Imputed via Train-only `Suburb×Type` median + `YearBuilt_was_imputed = 1`. |
| **`BuildingArea` (Zero / Micro / Extreme Area)** | `BuildingArea < 15 sqm` or `> 1,500 sqm` | **167** | **Masked to `NaN`** → Imputed via 5-tier Train-only hierarchical median + `BuildingArea_was_imputed = 1`. |
| **`Landsize` (Zero-Land Houses)** | `Landsize <= 0 sqm` and `Type == 'h'` | **222** | **Masked to `NaN`** → Imputed via 4-tier Train-only hierarchical median + `Landsize_was_imputed = 1`. |
| **`Landsize` (Extreme Acreage Cap)** | `Landsize > 15,000 sqm` | **36** | **Winsorized (capped)** at `15,000 sqm`. |
| **`Bedroom2` (Invalid Bedroom Count)** | `Bedroom2 <= 0` or `Bedroom2 > 10` | **21** | **Corrected** using `Rooms` count clipped to `[1, 10]`. |
| **`Bathroom` (Invalid Bathroom Count)** | `Bathroom <= 0` or `Bathroom > 8` | **48** | **Masked to `NaN`** → Imputed via Train-only `(Type, Rooms)` median. |
| **`Car` (Invalid Car Spaces)** | `Car < 0` or `Car > 10` | **5** | **Masked to `NaN`** → Imputed via Train-only `(Type, Rooms)` median. |
| **Total Rules Triggered** | **10 Explicit Domain Rules** | **561** | **`504` physical anomalies fixed/imputed + `57` extreme price outliers dropped.** |

---

## Table 3: Column-by-Column Missing Value Audit (Before vs. After Cleaning)

| Column | Present in `LESS`? | Raw `FULL` Missing (`n = 34,857`) | Pre-Imputation Priced Sales (`n = 27,192`) | Post-Cleaning Priced Sales | Post-Cleaning Unseen (`n = 7,607`) | Train-Only Imputation & Encoding Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`BuildingArea`** | No | 21,115 (`60.58%`) | 16,701 (`61.42%`) | **0** | **0** | 5-tier Train-only median (`Suburb×Type×Rooms → Suburb×Type → Region×Type×Rooms → Type×Rooms → Type`) + `BuildingArea_was_imputed` flag |
| **`YearBuilt`** | No | 19,306 (`55.39%`) | 15,140 (`55.68%`) | **0** | **0** | 3-tier Train-only median (`Suburb×Type → Suburb → Region`) + `YearBuilt_was_imputed` flag |
| **`Landsize`** | No | 11,810 (`33.88%`) | 9,415 (`34.62%`) | **0** | **0** | 4-tier Train-only median (`Suburb×Type×Rooms → Suburb×Type → Region×Type → Type`) + `Landsize_was_imputed` flag |
| **`Car`** | No | 8,728 (`25.04%`) | 6,813 (`25.06%`) | **0** | **0** | Train-only `(Type, Rooms)` median fallback to Train global median |
| **`Bathroom`** | No | 8,226 (`23.60%`) | 6,469 (`23.79%`) | **0** | **0** | Train-only `(Type, Rooms)` median fallback to Train global median |
| **`Bedroom2`** | No | 8,217 (`23.57%`) | 6,428 (`23.64%`) | **0** | **0** | Missing/invalid filled with `Rooms` clipped to `[1, 10]` |
| **`Lattitude`** | No | 7,976 (`22.88%`) | 6,241 (`22.95%`) | **0** | **0** | 2-tier Train-only spatial median (`Suburb → Region → Melbourne CBD`) |
| **`Longtitude`** | No | 7,976 (`22.88%`) | 6,241 (`22.95%`) | **0** | **0** | 2-tier Train-only spatial median (`Suburb → Region → Melbourne CBD`) |
| **`Price` (Target)** | Yes | 7,610 (`21.83%`) | 0 (`0.00%`) | **0** | **7,607** | `2` backfilled from `LESS` via 7-key match; `7,607` unpriced routed to `Fact_UnseenProperty` |
| **`CouncilArea`** | Yes | 3 (`0.01%`) | 1 (`0.00%`) | **0** | **0** | Backfilled from `LESS` suburb lookup; fallback to Train mode |
| **`Regionname`** | Yes | 3 (`0.01%`) | 1 (`0.00%`) | **0** | **0** | Backfilled from `LESS` suburb lookup; fallback to Train mode |
| **`Propertycount`** | Yes | 3 (`0.01%`) | 1 (`0.00%`) | **0** | **0** | Backfilled from `LESS` suburb lookup; fallback to Train median |
| **`Distance`** | Yes | 1 (`0.00%`) | 1 (`0.00%`) | **0** | **0** | Backfilled from `LESS` suburb lookup |
| **`Postcode`** | Yes | 1 (`0.00%`) | 1 (`0.00%`) | **0** | **0** | Backfilled from `LESS` suburb lookup |
| **`Suburb`, `Address`, `Rooms`, `Type`, `Method`, `SellerG`, `Date`** | Yes | 0 (`0.00%`) | 0 (`0.00%`) | **0** | **0** | Complete in raw CSVs (`0` missing values) |

---

## Table 4: Model Performance Comparison — Before (Step 1 Leak-Free Baseline) vs. After (Step 2 Log-Target + OOF Target Encoding + `RandomizedSearchCV`)

*Evaluated on the unseen 20% Holdout Test Set ($n = 5,439$) in original AUD dollar scale.*

| Regression Model | Step 1 Holdout $R^2$ | **Step 2 Holdout $R^2$** ($\Delta$) | Step 1 Holdout MAE | **Step 2 Holdout MAE** ($\Delta$) | Step 1 Holdout RMSE | **Step 2 Holdout RMSE** ($\Delta$) | Step 1 Holdout MAPE | **Step 2 Holdout MAPE** ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regression (Champion)** | 81.11% | **81.88%** (`+0.77%`) | \$152,089 | **\$144,337** (`-$7,751`) | \$254,703 | **\$249,455** (`-$5,248`) | 14.52% | **13.37%** (`-1.15 pp`) |
| **Random Forest Regression** | 78.80% | **78.40%** (`-0.40%`) | \$160,643 | **\$158,062** (`-$2,581`) | \$269,830 | **\$272,413** (`+$2,583`) | 15.34% | **14.60%** (`-0.74 pp`) |
| **Linear Regression (Log-Linear)** | 67.88% | **74.19%** (`+6.31%`) | \$222,245 | **\$181,931** (`-$40,314`) | \$332,148 | **\$297,782** (`-$34,366`) | 23.96% | **17.04%** (`-6.92 pp`) |
| **Decision Tree Regression** | 71.46% | **72.58%** (`+1.12%`) | \$191,668 | **\$185,924** (`-$5,744`) | \$313,117 | **\$306,919** (`-$6,198`) | 18.24% | **17.35%** (`-0.89 pp`) |
| **Naive Mean Baseline** | -0.01% | **-0.01%** | \$437,766 | **\$437,766** | \$586,133 | **\$586,133** | 52.34% | **52.34%** |

---

## Table 5: Complete 5-Fold Cross-Validation (`mean ± std`, $n = 21,753$) & Holdout Test ($n = 5,439$) Benchmark with 95% Bootstrap CIs

| Model | 5-Fold CV $R^2$ (`mean ± std`) | Holdout Test $R^2$ `[95% CI]` | 5-Fold CV MAE (`mean ± std`) | Holdout Test MAE `[95% CI]` | 5-Fold CV RMSE (`mean ± std`) | Holdout Test RMSE `[95% CI]` | 5-Fold CV MAPE (`mean ± std`) | Holdout Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **82.39% ± 0.51%** | **81.88%** `[79.98%, 83.85%]` | **\$149,786 ± \$3,971** | **\$144,337** `[$139,425, $148,778]` | **\$253,874 ± \$6,016** | **\$249,455** `[$232,052, $264,331]` | **13.80% ± 0.32%** | **13.37%** |
| **Random Forest** | 80.26% ± 0.35% | 78.40% `[76.17%, 80.94%]` | \$159,637 ± \$3,384 | \$158,062 `[$152,355, $162,445]` | \$268,819 ± \$4,631 | \$272,413 `[$251,668, $289,390]` | 14.72% ± 0.33% | 14.60% |
| **Linear Regression** | 75.18% ± 1.07% | 74.19% `[72.52%, 76.39%]` | \$182,405 ± \$3,901 | \$181,931 `[$175,984, $186,509]` | \$301,270 ± \$5,018 | \$297,782 `[$278,589, $311,614]` | 17.00% ± 0.27% | 17.04% |
| **Decision Tree** | 74.59% ± 0.58% | 72.58% `[70.21%, 75.43%]` | \$186,530 ± \$3,055 | \$185,924 `[$179,675, $191,835]` | \$304,896 ± \$2,529 | \$306,919 `[$284,421, $324,304]` | 17.23% ± 0.32% | 17.35% |

### Tuned Hyperparameters (`RandomizedSearchCV`, 5-Fold CV)
- **XGBoost (`XGBRegressor` inside `TransformedTargetRegressor`)**: `n_estimators = 650`, `learning_rate = 0.035`, `max_depth = 7`, `min_child_weight = 4`, `subsample = 0.90`, `colsample_bytree = 0.75`, `reg_alpha = 2.0`, `reg_lambda = 4.0`.
- **Random Forest (`RandomForestRegressor` inside `TransformedTargetRegressor`)**: `n_estimators = 200`, `max_depth = 26`, `min_samples_split = 6`, `min_samples_leaf = 3`, `max_features = 0.65`.
- **Decision Tree (`DecisionTreeRegressor` inside `TransformedTargetRegressor`)**: `max_depth = 10`, `min_samples_split = 12`, `min_samples_leaf = 20`.

---

## Table 6: Split Conformal Prediction Interval Calibration ($n_{\text{cal}} = 21,753$ OOF, $n_{\text{test}} = 5,439$ Holdout)

Nonconformity scores $s_i = |\log(1+y_i) - \log(1+\hat{y}_i^{\text{OOF}})|$ are stratified by dwelling type $t \in \{\text{h}, \text{t}, \text{u}\}$ to produce multiplicative intervals $[\exp(\log(1+\hat{y}) - \hat{q}_{1-\alpha}^{(t)}) - 1,\; \exp(\log(1+\hat{y}) + \hat{q}_{1-\alpha}^{(t)}) - 1]$.

| Dwelling Type | 95% Log Quantile ($\hat{q}_{0.95}^{(t)}$) | 80% Log Quantile ($\hat{q}_{0.80}^{(t)}$) | Equivalent Multiplicative Band (95%) | Holdout Empirical Coverage (95% Nominal) | Holdout Empirical Coverage (80% Nominal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **House (`h`)** | `0.3617` | `0.2037` | `[0.696 × ŷ, 1.436 × ŷ]` | — | — |
| **Townhouse (`t`)** | `0.3464` | `0.1993` | `[0.707 × ŷ, 1.414 × ŷ]` | — | — |
| **Unit / Apartment (`u`)** | `0.4182` | `0.2451` | `[0.658 × ŷ, 1.519 × ŷ]` | — | — |
| **Overall Holdout Test ($n = 5,439$)** | `0.3747` (Global) | `0.2119` (Global) | Dwelling-Type Stratified | **95.26%** | **81.30%** |

---

## Table 7: Top 15 Global Feature Importances (XGBoost Champion vs. Random Forest)

| Rank | XGBoost Feature | XGBoost Importance (%) | Random Forest Feature | Random Forest Importance (%) |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `PropertyType` (`h`, `t`, `u`) | **33.50%** | `SuburbTypeRoomsTargetEncoded` | **61.13%** |
| **2** | `SuburbTypeRoomsTargetEncoded` | **25.86%** | `SuburbTypeTargetEncoded` | **15.62%** |
| **3** | `SuburbTypeTargetEncoded` | **7.91%** | `SuburbTargetEncoded` | **3.46%** |
| **4** | `CouncilArea` | **5.49%** | `BuildingArea` | **2.06%** |
| **5** | `Regionname` | **4.33%** | `PriorSuburbMedianPrice` | **1.78%** |
| **6** | `Rooms` | **3.02%** | `Landsize` | **1.72%** |
| **7** | `TotalUtilityRooms` | **2.69%** | `YearBuilt` | **1.26%** |
| **8** | `SuburbTargetEncoded` | **2.21%** | `RadialDistanceCBD` | **1.20%** |
| **9** | `SaleMethod` | **1.93%** | `PropertyAge` | **0.99%** |
| **10** | `YearBuilt` | **1.84%** | `Rooms` | **0.95%** |
| **11** | `Bathroom` | **1.69%** | `Longtitude` | **0.88%** |
| **12** | `Bedroom2` | **1.30%** | `SuburbPremiumRatio` | **0.88%** |
| **13** | `BuildingArea` | **0.95%** | `FloorToLandRatio` | **0.82%** |
| **14** | `PropertyAge` | **0.78%** | `Lattitude` | **0.81%** |
| **15** | `Distance` | **0.73%** | `TotalUtilityRooms` | **0.80%** |

---

## Table 8: Holdout Residual & Error Slice Analysis by Price Band ($n = 5,439$, XGBoost Champion)

| Price Band (Actual Sale Price) | Holdout Sales ($n$) | Share of Test Set | Mean Actual Price | Mean Predicted Price | Mean Residual ($y - \hat{y}$) | Holdout MAE | Holdout RMSE | Holdout MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Under \$600k** | 1,167 | 21.46% | \$479,913 | \$523,339 | -\$43,427 | **\$66,536** | \$98,404 | 14.68% |
| **\$600k – \$1.0M** | 2,152 | 39.57% | \$785,403 | \$813,955 | -\$28,552 | **\$96,365** | \$143,164 | **12.16%** |
| **\$1.0M – \$1.5M** | 1,194 | 21.95% | \$1,235,147 | \$1,214,515 | +\$20,632 | **\$156,194** | \$208,557 | **12.61%** |
| **\$1.5M – \$2.5M** | 775 | 14.25% | \$1,866,012 | \$1,745,213 | +\$120,799 | **\$272,102** | \$362,894 | 14.56% |
| **Over \$2.5M** | 151 | 2.78% | \$3,234,043 | \$2,613,542 | +\$620,501 | **\$679,815** | \$898,904 | **20.27%** |

---

## Table 9: Holdout Error Slice Analysis by Dwelling Type & Metropolitan Region ($n = 5,439$)

### A. By Dwelling Type (`Dim_Property.PropertyType`)

| Dwelling Type | Code | Holdout Sales ($n$) | Median Actual Price | Median Predicted Price | Mean Residual ($y - \hat{y}$) | Holdout $R^2$ | Holdout MAE | Holdout RMSE | Holdout MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **House / Cottage / Villa** | `h` | 3,666 | \$1,007,000 | \$1,034,812 | +\$22,206 | **79.57%** | \$162,389 | \$279,133 | **12.95%** |
| **Townhouse** | `t` | 569 | \$850,000 | \$870,402 | +\$5,152 | **76.59%** | \$123,332 | \$186,917 | **12.67%** |
| **Unit / Duplex / Apartment** | `u` | 1,204 | \$575,000 | \$597,282 | +\$7,138 | **67.12%** | **\$99,300** | \$165,405 | 14.95% |

### B. By Metropolitan Region (`Dim_Location.Region`)

| Metropolitan Region | Holdout Sales ($n$) | Median Actual Price | Median Predicted Price | Mean Residual ($y - \hat{y}$) | Holdout $R^2$ | Holdout MAE | Holdout RMSE | Holdout MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Southern Metropolitan** | 1,705 | \$1,208,000 | \$1,212,530 | +\$18,860 | **81.32%** | \$202,602 | \$319,565 | 15.50% |
| **Northern Metropolitan** | 1,599 | \$760,000 | \$757,563 | +\$18,333 | **76.62%** | \$121,277 | \$224,788 | 12.81% |
| **Western Metropolitan** | 1,151 | \$760,000 | \$758,832 | +\$14,567 | **79.00%** | \$105,009 | \$170,310 | 12.10% |
| **Eastern Metropolitan** | 633 | \$1,030,000 | \$1,045,489 | +\$20,834 | **71.20%** | \$144,734 | \$256,689 | 12.54% |
| **South-Eastern Metropolitan** | 269 | \$800,000 | \$797,962 | +\$2,995 | **71.09%** | \$101,377 | \$168,165 | 11.05% |
| **Eastern Victoria** | 39 | \$650,000 | \$662,267 | -\$3,069 | 66.29% | \$67,310 | \$91,354 | 10.88% |
| **Northern Victoria** | 31 | \$601,000 | \$607,683 | +\$35,252 | 73.05% | \$88,957 | \$123,615 | 11.71% |
| **Western Victoria** | 12 | \$400,000 | \$403,232 | -\$22,460 | -17.50% (`n=12`) | \$46,401 | \$59,287 | 13.64% |

