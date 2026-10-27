# COMP8811 Assignment 2 — Limitations, Critical Reflection & Future Work

This document provides a critical academic reflection on the predictive modeling, star schema architecture, and business intelligence implementation for the Melbourne residential housing dataset, directly addressing rubric requirements on critical evaluation and honest appraisal of machine learning limitations.

---

## 1. Temporal Constraints: The 2016–2018 Market Snapshot

### Empirical Reality
The underlying dataset spans transactions from **January 2016 through October 2018** (spanning 112 discrete settlement dates). During this 34-month window:
- The Reserve Bank of Australia (RBA) cash rate remained virtually static, easing modestly from **2.00% in Q1 2016 down to 1.50% by August 2016**, where it remained locked through 2018.
- Because the macro interest rate exhibits near-zero longitudinal variance ($\sigma = 0.16\%$) over this period, statistical regression algorithms cannot identify an empirical elasticity for interest rates without significant collinearity with temporal trends.

### Mitigation & Honest System Design
- Rather than artificially multiplying predictions by interest rate adjustments (which introduced post-hoc bias in earlier iterations), the platform treats `InterestRate` strictly as an **external scenario indicator** in `Dim_Market` and clearly labels the input on Tab 1 as:
  > *"Interest Rate Scenario (Display Only · Not a Model Feature)"*
- Macro price drift is captured via chronological, strictly past-only regional medians (`PriorPeriodMedianPrice`) and transaction volumes (`QuarterTransactionVolume`) rather than uncalibrated interest rate multipliers.

---

## 2. Missing Data Mechanisms: High Missingness in Physical Attributes

### Empirical Missingness
Prior to data cleaning, significant portions of key physical attributes were absent in `melbourne_housing_full.csv`:
- **`BuildingArea`**: **60.58% missing** ($21,115$ of $34,857$ raw listings; $16,701$ of $27,192$ validated sales).
- **`YearBuilt`**: **55.39% missing** ($19,306$ raw listings; $15,140$ validated sales).
- **`Landsize`**: **33.88% missing** ($11,810$ raw listings; $9,415$ validated sales).
- **`Bathroom` & `Car`**: **23.60%** and **25.04% missing** respectively.

### Structural Bias & Imputation Limitations
1. **Missing Not at Random (MNAR)**: In Australian real estate listings, missing floor areas and construction years are rarely random; older homes without formal floor plans or unrenovated dwellings are substantially more likely to omit building area and construction era than newly developed master-planned estates.
2. **5-Tier Hierarchical Imputation**: To avoid collapsing missing floor areas to an arbitrary global constant (such as the naive $96.5\text{ m}^2$ median observed in early iterations), we introduced a **5-tier hierarchical median imputer fitted exclusively on the 80% training set**:
   $$\text{Suburb} \times \text{Type} \times \text{Rooms} \longrightarrow \text{Suburb} \times \text{Type} \longrightarrow \text{Region} \times \text{Type} \times \text{Rooms} \longrightarrow \text{Type} \times \text{Rooms} \longrightarrow \text{Type}$$
3. **Explicit Missingness Indicators**: To prevent tree and linear models from assuming imputed values reflect true physical measurements, deterministic binary indicator flags (`BuildingArea_was_imputed`, `YearBuilt_was_imputed`, `Landsize_was_imputed`) were engineered prior to imputation. While this allows the tree models to learn separate partition paths for imputed records, imputing over 60% of floor areas inevitably suppresses natural physical variance.

---

## 3. Unobserved Hedonic Quality Drivers in Real Estate

Real property pricing is driven by hedonic attributes that are entirely omitted from standard tabular MLS/council scrape data:
1. **Internal Renovation & Finish Grade**: Two Victorian cottages in Richmond with identical 3-bedroom, 1-bathroom specifications and $180\text{ m}^2$ land area can differ in market clearance price by over $\$500,000$ depending on whether the property features an unrenovated 1970s interior or a designer architect-led renovation with premium marble finishes.
2. **Topography & Views**: Slope, elevation, natural sunlight orientation (north-facing rear yards in the Southern Hemisphere), and city skyline or bay views carry substantial valuation premia that cannot be captured by latitude/longitude alone.
3. **School Catchment Zones**: In Melbourne, public school zones (e.g., Balwyn High School, University High School, McKinnon Secondary College) command verified real estate premia between $\$100,000$ and $\$250,000$ across adjacent streets that share the same postal code.
4. **Planning Overlays & Heritage Protections**: Local Council heritage overlays restrict redevelopment and second-story extensions, while multi-dwelling zoning allowances significantly inflate land value for developers.

---

## 4. Heteroscedasticity & Severe Under-Prediction in the Luxury Tail

### Empirical Residual Finding
Residual analysis across the 5,439 Holdout Test sales reveals pronounced heteroscedasticity:

| Price Band | Holdout Count ($n$) | Holdout MAE | Holdout MAPE | Mean Residual ($y - \hat{y}$) | Median Absolute Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Under \$600k** | 1,167 | \$66,536 | 14.68% | -\$43,427 | \$46,356 |
| **\$600k – \$1.0M** | 2,152 | \$96,365 | **12.16%** | -\$28,552 | \$72,068 |
| **\$1.0M – \$1.5M** | 1,194 | \$156,194 | **12.61%** | +\$20,632 | \$119,658 |
| **\$1.5M – \$2.5M** | 775 | \$272,102 | 14.56% | +\$120,799 | \$217,315 |
| **Over \$2.5M** | 151 | **\$679,815** | **20.27%** | **+\$620,501** | \$497,462 |

### Root Cause Analysis
- **Core Market ($600k–$1.5M)**: Represents **61.5% of sales**, where the model achieves outstanding predictive calibration with MAPEs between **12.16% and 12.61%**.
- **Luxury Tail ($> $2.5M)**: The model suffers a severe systematic under-prediction bias (mean residual **$+\$620,501$**; MAE **$\$679,815$**). Because luxury properties represent less than 3% of training volume, standard squared-error loss and log-transformed regression optimize for majority density. Prestige estates command private treaty negotiated prices driven by trophy provenance, architectural uniqueness, and deep-pocketed emotional bidding that cannot be linearly extrapolated from room counts.

---

## 5. Concrete Roadmap for Future Work

If extending this system into an institutional-grade automated valuation model (AVM), four concrete engineering upgrades should be prioritized:

1. **Computer Vision & Multimodal Image Embeddings**:
   - Ingest listing photographic streams (facade, kitchen, bathrooms, aerial block view) and extract latent quality features via a fine-tuned Vision Transformer (e.g., CLIP / ResNet-50).
   - This directly quantifies interior condition grade and natural light, closing the unobserved renovation gap.

2. **Geospatial & Vector Enrichment**:
   - Ingest Victorian Department of Education school zone boundary shapefiles (`.geojson`) and perform point-in-polygon spatial joins.
   - Calculate road network distance to closest train stations, tram corridors, public parks, and high-street retail precincts via OpenStreetMap (OSM) routing.

3. **Two-Stage or Quantile Luxury Regression**:
   - Train a specialized high-net-worth tier model or employ **Gradient Boosted Quantile Regression** ($\tau \in \{0.10, 0.50, 0.90\}$) to provide asymmetric risk distributions specifically tailored to high-variance prestige properties.

4. **Live Pipeline Ingestion & Drift Monitoring**:
   - Transition from batch SQLite to an automated event-driven pipeline (e.g., Apache Airflow + PostgreSQL) that monitors data drift, concept drift, and recalculates Split Conformal quantiles as rolling macroeconomic conditions evolve.

