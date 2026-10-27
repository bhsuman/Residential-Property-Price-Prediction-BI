# COMP8811 Assignment 2 — UI Screenshots Checklist & Ready-to-Paste Report Captions

Use this checklist while running the application at **http://127.0.0.1:5000** (recommended browser zoom: `90%` or `100%` at `1920×1080` resolution) to capture high-impact evidence figures for your COMP8811 Assignment 2 report and presentation slides.

---

## Screenshot 1 — Tab 1: Live Predictor with Dynamic Local TreeSHAP (`Richmond 3 Bed House`)
- **Where to navigate**: Tab 1 (**Property Valuation (Live Predictor)**).
- **Exact interaction**:
  1. Click the **`Richmond 3 Bed House`** quick-fill button under *Sample Profiles*.
  2. Hover over or ensure the **95% Split Conformal Prediction Interval** info icon (`ⓘ`) is visible under the main price (`$1,856,187`).
  3. Ensure the bottom-left chart toggle is set to **`Local TreeSHAP`** and the bottom-right **Cross-Model Valuation Comparison** chart is visible.
- **What this proves to the marker**:
  - 1-to-1 alignment with the Assignment 1 hedonic features (`Dim_Property`, `Dim_Location`, `Dim_Market`).
  - Honest labeling of the **Interest Rate Scenario (Display Only · Not a Model Feature)**.
  - Exact per-prediction TreeSHAP contributions (`Suburb & Location: +$884,193 [90.7%]`, `Rooms & Layout: +$28,580 [2.9%]`, etc.) summing from the training baseline (`$908,869`) to the predicted price (`$1,856,187`).
  - Dwelling-type stratified **Split Conformal Prediction Interval** (`95.26%` empirical holdout coverage).
- **Ready-to-paste report caption**:
  > **Figure 1: Interactive Property Valuation Predictor (Tab 1) — Richmond 3-Bedroom House Profile.** The interface maps user inputs directly to `Dim_Property` and `Dim_Location` attributes, generating a real-time XGBoost Champion valuation ($\hat{y} = \$1,856,187$, 95% Split Conformal Interval $\$1,291,683\text{--}\$2,667,385$), dynamic instance-level TreeSHAP hedonic value contributions relative to the training baseline ($\$908,869$), and cross-model predictions across XGBoost, Random Forest, Linear Regression, and Decision Tree.

---

## Screenshot 2 — Tab 1: Contrasting Sample Profile Showing Dynamic TreeSHAP Shift (`South Yarra 2 Bed Unit`)
- **Where to navigate**: Tab 1 (**Property Valuation (Live Predictor)**).
- **Exact interaction**:
  1. Click the **`South Yarra 2 Bed Unit`** quick-fill button.
  2. Observe how the **Hedonic Value Composition (Local TreeSHAP)** bars and percentages change dynamically (e.g., `Rooms & Layout` becomes negative `-$225,650 [56.7%]` and `Building & Floor Area` shifts to `-$49,337 [12.4%]` relative to the baseline because it is a compact 2-bedroom apartment).
  3. Also capture a second sub-screenshot with the **`Global Importance`** toggle clicked on the same chart.
- **What this proves to the marker**:
  - Proves the Hedonic Value Composition chart is genuinely driven by per-instance XGBoost TreeSHAP (`pred_contribs=True`) rather than static weights.
- **Ready-to-paste report caption**:
  > **Figure 2: Dynamic Instance-Level TreeSHAP Sensitivity (Tab 1) — South Yarra 2-Bedroom Unit Profile.** Switching from a 3-bedroom Richmond house to a 2-bedroom South Yarra unit dynamically updates the TreeSHAP decomposition, revealing negative structural adjustments for compact apartment typology (`Rooms & Layout: -$225,650`, `Building & Floor Area: -$49,337`) alongside positive inner-suburb location value (`Suburb & Location: +$78,242`).

---

## Screenshot 3 — Tab 1 (Bottom): Out-of-Sample Comparable Settled Sales Cards
- **Where to navigate**: Scroll to the bottom of Tab 1 (**Comparable Settled Sales (Out-of-Sample Predictions)**).
- **Exact interaction**:
  1. With `Richmond 3 Bed House` or `Kew 4 Bed Residence` active, capture the 8 comparable property cards.
  2. Point out the **`Holdout Test`** and **`5-Fold OOF`** badges on each card.
- **What this proves to the marker**:
  - Comparable historical sales are strictly scored out-of-sample (Holdout Test prioritized, 5-fold Out-of-Fold fallback), eliminating in-sample memorization.
- **Ready-to-paste report caption**:
  > **Figure 3: Out-of-Sample Comparable Settled Sales Retrieval (Tab 1).** Querying `Fact_PropertySale` retrieves settled transactions matching the target suburb, dwelling type, and bedroom profile, displaying strictly out-of-sample predictions (`Holdout Test` and `5-Fold OOF` badges) alongside actual settlement prices.

---

## Screenshot 4 — Tab 2: BI Decision Support Dashboard (Default Holdout Test Split = 5,439 Sales)
- **Where to navigate**: Click Tab 2 (**BI Decision Support Dashboard**).
- **Exact interaction**:
  1. Keep **Evaluation Split (Train vs Test)** on its default: **`Holdout Test Only (20% · 5,439)`**.
  2. Capture the top filter bar (with the green badge `Holdout Test Set Only (5,439 Sales · Zero In-Sample Leakage)`), the 4 KPI cards (`5,439` Holdout Sales, Mean Actual `$1,034,745`, Mean Predicted `$1,017,658`, Holdout MAE `$144,337`, `R² = 81.9% | RMSE = $249,455 | MAPE = 13.4%`), and the 2×2 Plotly analytical grid.
- **What this proves to the marker**:
  - Eliminates the previous in-sample leakage where all 27,192 rows were averaged together.
  - Demonstrates interactive Star Schema OLAP slicing across `Fact_PropertySale`, `Dim_Property`, `Dim_Location`, `Dim_Date`, and `Dim_Market`.
- **Ready-to-paste report caption**:
  > **Figure 4: Star Schema BI Decision Support Dashboard on the Holdout Test Split (Tab 2, $n = 5,439$).** All KPI cards and analytical charts default strictly to the unseen 20% holdout test partition ($\text{MAE} = \$144,337$, $\text{RMSE} = \$249,455$, $R^2 = 81.88\%$, $\text{MAPE} = 13.37\%$), comparing actual vs. predicted prices across metropolitan regions, quarterly settlement trends, the $45^\circ$ parity line, and dwelling-type/bedroom segments.

---

## Screenshot 5 — Tab 2 (Bottom): Unseen Properties Scored (`Fact_UnseenProperty` — 7,607 Listings) Compared Against Same-Type Suburb Median
- **Where to navigate**: Scroll to the bottom table of Tab 2 (**Unseen Properties Scored (`Fact_UnseenProperty` — 7,607 Listings) & Relative Type Comparison**).
- **Exact interaction**:
  1. Capture the default view (`Sort: Below Typical Price for Type`) showing 15 distinct suburbs.
  2. Highlight the right-hand column showing **`Predicted Price (vs. Same-Type Median)`** (e.g., `Suburb Unit Med` for units and `Suburb House Med` for houses) and the honest relative badges (`Below Typical for Type`, `Typical Range for Type`, `Above Typical for Type`).
- **What this proves to the marker**:
  - Fixes the misleading comparison of 1-bed units against all-dwelling suburb house medians and replaces speculative "Value Opportunity" labels with honest same-suburb, same-dwelling-type benchmarks.
- **Ready-to-paste report caption**:
  > **Figure 5: Out-of-Sample Valuation Screening of 7,607 Unpriced Listings (`Fact_UnseenProperty`, Tab 2).** Unpriced Melbourne listings are scored by the tuned XGBoost Champion model and benchmarked against the training-set median price for the **same suburb and dwelling type** (`SuburbTypeMedianPrice`), preventing cross-typology distortion between apartments and detached houses.

---

## Screenshot 6 — Tab 3 (Top & Middle): 4-Model Benchmark Cards, CV ± SD Error Bars & Holdout Residual Analysis
- **Where to navigate**: Click Tab 3 (**Models & Star Schema Pipeline**).
- **Exact interaction**:
  1. Capture **Section 1** (the 4 model cards showing `Holdout Test R²`, `5-Fold CV R² (Mean ± SD)`, `Holdout Test MAE`, `Holdout Test RMSE`, `Holdout Test MAPE`, and `5-Fold CV MAE (± SD)`, plus the 3 benchmark charts below them).
  2. Capture **Section 2** (**Holdout Residual & Error Slice Analysis**), including the amber empirical finding callout box and the 3 diagnostic charts (`Residuals vs. Predicted Price`, `Holdout Error by Price Band — MAE & MAPE`, and `Holdout Error by Region & Dwelling Type`).
- **What this proves to the marker**:
  - Rigorous model comparison with 5-fold cross-validation (`mean ± std`) and holdout metrics (`R²`, `MAE`, `RMSE`, `MAPE`), plus transparent residual diagnostics exposing luxury-tier heteroscedasticity (`Over $2.5M` `MAE = $679,815`, `MAPE = 20.27%` vs. `$600k–$1.5M` `MAPE = 12.16%–12.61%`).
- **Ready-to-paste report caption**:
  > **Figure 6: Multi-Model Regression Benchmark and Holdout Residual Diagnostics (Tab 3).** Section 1 compares 5-fold cross-validation (`mean ± SD`) and 20% holdout test performance across XGBoost ($R^2 = 81.88\%$, $\text{MAE} = \$144,337$, $\text{MAPE} = 13.37\%$), Random Forest ($78.40\%$), Linear Regression ($74.19\%$), and Decision Tree ($72.58\%$). Section 2 diagnoses holdout residuals across price bands, regions, and dwelling types, highlighting strong calibration in the $\$600\text{k--}\$1.5\text{M}$ core market ($\text{MAPE} = 12.16\%\text{--}12.61\%$) and positive heteroscedastic tail errors above $\$2.5\text{M}$.

---

## Screenshot 7 — Tab 3 (Bottom): 4-Panel ETL Data Quality Audit, Downloadable CSV/JSON Buttons & Star Schema Table Inspector
- **Where to navigate**: Scroll to **Section 3** of Tab 3 (**3. ETL Data Cleaning Audit & Dimensional Star Schema Warehouse**).
- **Exact interaction**:
  1. Capture the top-right **`Download Data Quality Summary (CSV)`** and **`Download Audit Report (JSON)`** buttons, the 6 summary cards (`97,880` Source Records, `3 Dups / 34,855 Merged`, `27,192` Validated Sales, `7,607` Unpriced Appraised, `504 Fixed + 57 Dropped`, `21,753 / 5,439` Split), the 4 audit tables (**Panels A, B, C, D**), and the interactive Star Schema table browser below.
- **What this proves to the marker**:
  - Complete transparency and reproducibility of the dual-CSV data cleaning pipeline, train-only hierarchical imputation, explicit outlier thresholds, and leak-free feature encoding.
- **Ready-to-paste report caption**:
  > **Figure 7: End-to-End ETL Data Quality Audit and Star Schema Warehouse Inspector (Tab 3).** Panels A–D report exact record dispositions from 97,880 raw rows to 27,192 validated sales and 7,607 unseen listings, 10 explicit outlier and anomaly rules (504 physical fixes, 57 extreme price tail drops), column-level missingness before and after Train-only hierarchical imputation (0 remaining feature nulls), and the 31-to-76 dimensional encoding scheme, with one-click CSV and JSON audit exports.

