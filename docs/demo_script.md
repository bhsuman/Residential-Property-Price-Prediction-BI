# COMP8811 Assignment 2 — 5 to 7 Minute Live Demo Script & Marker Q&A Guide

**Project Title:** Residential Property Price Prediction & BI Decision Support  
**Case Study:** Melbourne Housing Dataset (Bayleys Real Estate Advisory)  
**URL:** `http://127.0.0.1:5000` (Launch via `python run.py`)

---

## Timing Breakdown (Total: 6–7 Minutes)

| Section | Target Duration | Navigation Focus | Key Talking Points |
| :--- | :---: | :--- | :--- |
| **1. Introduction & Architecture** | 1.0 min | Global Header & Tab 1 | Project scope, dual-CSV ingestion, star schema, zero-leakage framework |
| **2. Tab 1: Live Valuation Predictor** | 2.5 min | Tab 1 (Predictor) | Sample profiles, dynamic TreeSHAP, conformal prediction intervals, comps |
| **3. Tab 2: BI Decision Support** | 1.5 min | Tab 2 (Dashboard) | Default holdout test split, OLAP slicers, parity plot, unseen listings |
| **4. Tab 3: Models & Pipeline Audit** | 1.5 min | Tab 3 (Models/ETL) | 4-model benchmark, CV ± SD, residual heteroscedasticity, ETL audit & CSV download |
| **5. Marker Q&A Defense** | Open | Standby on Tab 1/2/3 | Data leakage, TreeSHAP mechanics, luxury error, conformal guarantees |

---

## Minute-by-Minute Presentation Script

### 0:00 – 1:00 | Project Overview & System Framing
> *"Good morning / afternoon. Today I am presenting my COMP8811 Assignment 2 project: an end-to-end Machine Learning and Business Intelligence platform for residential property price prediction in metropolitan Melbourne.*
>
> *Our goal is to assist property advisory firms—such as Bayleys—in navigating pricing uncertainty. The platform unifies **97,880 raw transaction records** across two CSV sources into a **5-table SQLite Dimensional Star Schema**, trains and compares **4 regression models**, and serves both an interactive property valuation engine and an executive OLAP decision support dashboard.*
>
> *Crucially, every component has been engineered to strictly prevent data leakage: all missing-value imputation, spatial target encoding, and scaler transformations are fitted exclusively on the 80% training partition."*

---

### 1:00 – 3:30 | Tab 1: Live Property Valuation Predictor
*(Action: Click the `Richmond 3 Bed House` quick-fill button at the top of Tab 1)*

> *"Here on Tab 1—our Live Predictor—the left specification panel maps 1-to-1 with our Assignment 1 Star Schema dimensions (`Dim_Property`, `Dim_Location`, and `Dim_Market`). Notice that our interest rate field is honestly labeled as a **scenario display**, because across 2016 to 2018 the RBA rate remained essentially flat, so we do not artificially inject uncalibrated multipliers.*
>
> *Looking at our valuation card on the right:*
> - *Our **XGBoost Champion Model** predicts **$1,856,187**.*
> - *Right underneath, we display a mathematically grounded **Split Conformal Prediction Interval** of **$1,291,683 to $2,667,385**. Unlike arbitrary heuristic percentage bands, this interval is calibrated on 21,753 held-out out-of-fold log-residuals stratified by dwelling type, providing a proven **95.26% empirical coverage guarantee** on our test set.*
> - *Directly below, our four academic and BI metrics display the prior-period suburb median, the calculated property age and floor-to-land ratio, the radial CBD distance, and our champion test error margin ($R^2 = 81.88\%$, $\text{MAE} = \$144,337$).*

*(Action: Point to the bottom-left chart 'Hedonic Value Composition')*

> *"Now examine our **Hedonic Value Composition** chart. In our early iterations, this chart displayed static percentage weights. Today, it is powered by **exact instance-level TreeSHAP feature contributions** (`pred_contribs=True`). Starting from the training baseline of **$908,869**, we see that location contributes **+$884,193**, layout contributes **+$28,580**, and land size adds **+$26,589**, summing precisely to our final predicted valuation."*

*(Action: Click `South Yarra 2 Bed Unit`)*

> *"Watch what happens when I switch to a 2-bedroom South Yarra apartment. The TreeSHAP drivers dynamically shift: `Rooms & Layout` and `Floor Area` now exert substantial **negative deductions** relative to the average Melbourne dwelling, while inner-city location remains a positive driver. We can also toggle between local TreeSHAP and global feature importance.*
>
> *At the bottom of Tab 1, our **Comparable Settled Sales** section queries `Fact_PropertySale` and displays historical sales matching this profile, marked with explicit **`Holdout Test`** and **`5-Fold OOF`** badges to ensure zero in-sample retrieval."*

---

### 3:30 – 5:00 | Tab 2: BI Decision Support Dashboard
*(Action: Click Tab 2 'BI Decision Support Dashboard')*

> *"Switching to Tab 2, we arrive at our executive Business Intelligence dashboard. A critical flaw in naive real estate dashboards is in-sample leakage—reporting training rows alongside test rows.*
>
> *Notice our top filter: it **defaults strictly to our 5,439 Holdout Test sales**, verified by the green badge. Our 4 KPI cards show an actual test mean of **$1,034,745** and a predicted mean of **$1,017,658**, achieving a realistic Holdout MAE of **$144,337** and MAPE of **13.37%**.*
>
> *Our 2×2 analytical grid provides interactive OLAP cross-filtering: regional actual vs. predicted bars, chronological quarterly trajectories, our 45° parity scatter plot, and a granular error breakdown across property types and room counts.*
>
> *At the bottom, our **Unseen Properties Scored** table screens all 7,607 unpriced listings from `Fact_UnseenProperty`. Crucially, rather than comparing a 1-bed flat against a multi-million-dollar all-dwelling suburb median, our engine benchmarks each prediction against the training median for the **same suburb and same dwelling type**, providing honest categorization: 'Below Typical for Type' or 'Above Typical for Type'."*

---

### 5:00 – 6:30 | Tab 3: Models & Star Schema Pipeline
*(Action: Click Tab 3 'Models & Star Schema Pipeline')*

> *"Finally, Tab 3 provides complete transparency into our modeling, calibration, and data warehouse pipeline.*
>
> *In Section 1, our benchmark cards display 5-fold cross-validation means and standard deviations alongside our holdout test results across all 4 models:*
> - *Linear Regression achieved an $R^2$ of **74.19%** and MAE of **$181,931** after log-target transformation linearized price effects.*
> - *Decision Tree reached **72.58%**.*
> - *Random Forest reached **78.40%** ($R^2_{\text{CV}} = 80.26\% \pm 0.35\%$).*
> - *And our champion **XGBoost** model achieved **81.88% holdout $R^2$**, **$144,337 MAE**, and **13.37% MAPE** ($R^2_{\text{CV}} = 82.39\% \pm 0.51\%$).*
>
> *In Section 2, our **Holdout Residual & Error Slice Analysis** diagnoses model limitations: predictions are exceptionally well-calibrated for properties between $600k and $1.5M (MAPEs between 12.16% and 12.61%). However, above $2.5M, heteroscedasticity increases (MAE $679,815) because unobserved prestige factors like luxury renovations and trophy bidding are not captured in tabular MLS data.*
>
> *In Section 3, our ETL audit panel details every duplicate removed, our 10 explicit outlier rules (504 physical anomalies fixed, 57 extreme price outliers dropped), and column-by-column missingness before and after cleaning. Markers can download our full data quality summary in either CSV or JSON with a single click, and browse all 5 dimensional Star Schema tables directly in the UI.*
>
> *Thank you, and I welcome any questions."*

---

## Marker Q&A Defense Guide

### Question 1: "Why did you use log-transformation on the target variable ($y = \text{Price}$)?"
**Strong Answer:**
> *"Melbourne property prices are strongly right-skewed, ranging from $120,000 to over $5,000,000 with a heavy positive tail. Training models directly on dollar prices penalizes high-value errors quadratically, causing linear and tree models to over-fit luxury outliers while neglecting the core market.*
>
> *By wrapping our models in `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`, errors are minimized in relative percentage terms ($\log(1+y)$). This stabilized training, improved Linear Regression holdout $R^2$ by +6.31% and reduced MAE by over $40,000$, and reduced XGBoost holdout MAE from $152,089 down to $144,337."*

### Question 2: "How did you prevent data leakage during feature engineering and target encoding?"
**Strong Answer:**
> *"We enforced strict chronological and split-based firewalls:*
> 1. *The 80/20 train/test split was performed **before** fitting any missing-value imputer, scaler, or encoder.*
> 2. *Our hierarchical median imputers and historical suburb benchmarks were fitted **strictly on the 21,753 training records**.*
> 3. *For our high-cardinality Suburb target encoding, computing raw training averages would lead to massive target leakage. Therefore, we used a **5-fold out-of-fold Empirical Bayes scheme** within the training partition: fold $k$'s encodings were computed exclusively from the other 4 folds. Test and unseen records were transformed using the full-training encoder only.*
> 4. *Macro market medians used strictly past-only transactions occurring chronologically prior to each sale date."*

### Question 3: "What is Split Conformal Prediction, and why is it better than a standard $\pm 10\%$ band?"
**Strong Answer:**
> *"A fixed $\pm 10\%$ or $\pm 1.96 \times \text{RMSE}$ interval assumes homoscedastic, normally distributed residuals. In real estate, variance expands multiplicatively with price, and apartments have different uncertainty profiles than detached houses.*
>
> *We calibrated nonconformity scores $s_i = |\log(1+y_i) - \log(1+\hat{y}_i^{\text{OOF}})|$ on 21,753 held-out training residuals stratified by dwelling type (`h`, `t`, `u`). Taking the $(1-\alpha)$ quantile yields exact finite-sample, distribution-free coverage. When evaluated on our 5,439 unseen holdout test sales, the 95% nominal interval achieved an empirical test coverage of **95.26%**, proving mathematical validity."*

### Question 4: "Why does the model under-predict properties over $2.5M, and how would you fix it in production?"
**Strong Answer:**
> *"Luxury properties above $2.5M represent less than 3% of our dataset (151 of 5,439 test sales). At this prestige level, valuations are driven by unobserved hedonic features omitted from tabular council data—such as high-end designer renovations, panoramic city views, heritage architecture, and emotional auction bidding.*
>
> *In production, we would solve this by ingesting listing photographs via a Vision Transformer (CLIP/ResNet) to extract an interior finish quality score, integrating Victorian school catchment shapefiles via spatial joins, and training a specialized two-stage luxury quantile regression model."*

