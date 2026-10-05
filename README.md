# Specific Implementation Guide: BiciMAD Operations Optimization Thesis

This guide provides a concrete, end-to-end technical blueprint. It assumes the primary data source is the official historical package (2017–2023) published by EMT/Ayuntamiento de Madrid (ZIP files containing anonymized trips + station status), supplemented by the static stations dataset and (optionally) external weather/calendar data. The target is a predictive system that estimates, for each station and time window, the number of bikes needed to maximize **bike utilization rate**, with an optional linear-programming redistribution layer.

---

## 1. Data Pipeline Design

**Recommended architecture** (modular, reproducible, temporal-leakage-safe):

```
raw/                    # Immutable downloads (ZIPs + static stations)
├── trips/
├── status/
└── stations_static/
processed/
├── trips_clean.parquet
├── status_clean.parquet
├── stations_enriched.parquet
feature_store/
├── station_time_features.parquet   # Main modeling table (station × time-bin)
└── targets.parquet
models/
evaluation/
```

**Pipeline stages** (implement as Python modules or Prefect/Airflow tasks):

1. **Ingestion**
   - Download yearly ZIPs.
   - Extract trips (JSON/CSV) and station-status files.
   - Load static stations (CSV/GeoJSON) once.
   - Prefer Parquet + PyArrow or DuckDB for intermediate storage (columnar, efficient filtering).

2. **Parsing & Normalization**
   - Parse timestamps (`unplug_hourTime` → UTC-aware datetime).
   - Standardize station IDs (integer).
   - Convert `travel_time` to minutes; handle null tracks.
   - Align status snapshots to a common frequency (recommended: 15-min or 30-min bins; aggregate hourly if native resolution is coarser).

3. **Temporal Aggregation**
   - Create a regular time grid per station (e.g., 15-min bins).
   - From trips: count starts (unplugs), ends (plugs), net flow per bin.
   - From status: interpolate or forward-fill `dock_bikes`, `free_bases`, `total_bases`; compute occupancy ratio = `dock_bikes / total_bases`.
   - Join on `(station_id, time_bin)`.

4. **Enrichment**
   - Add calendar features, weather (if used), station metadata (capacity, coordinates, district).
   - Compute lagged and rolling statistics (see Feature Engineering).

5. **Split**
   - Strict temporal split: train ≤ 2021, validation 2022, test 2023 (or rolling-origin cross-validation).
   - Never shuffle across time.

6. **Performance considerations**
   - Process year-by-year or month-by-month to control memory.
   - Use Dask or Polars if full history exceeds RAM.
   - Log runtime, peak memory, and row counts at each stage (pipeline performance metric).
   - Make the pipeline idempotent and checkpointed so re-runs after schema changes are cheap.

**Output of the pipeline**: a single clean table `station_time_features` ready for modeling + a targets table.

---

## 2. Data Quality Assessment Method

Perform a systematic audit before and after cleaning. Document findings in a “Data Quality Report” chapter.

**Dimensions & concrete checks**:

| Dimension              | Concrete Metrics / Checks                                                                 | Action if Failing                          |
|------------------------|-------------------------------------------------------------------------------------------|--------------------------------------------|
| Completeness           | % missing timestamps, station IDs, travel_time, dock_bikes; stations with zero activity   | Drop or impute; flag low-activity stations |
| Validity               | travel_time ∈ [1 min, 4 h]; occupancy ∈ [0, capacity]; coordinates inside Madrid bbox     | Clip or remove outliers                    |
| Consistency            | Starts ≈ Ends at system level over long windows; total_bases stable per station           | Investigate large discrepancies            |
| Uniqueness             | Duplicate (user_day_code + timestamp + stations) rate                                     | Deduplicate                                |
| Timeliness / Coverage  | Gaps in status snapshots; stations appearing/disappearing                                 | Document service changes                   |
| Accuracy (proxy)       | Cross-check trip counts vs. official daily aggregates if available                        | Flag systematic bias                       |

**Implementation**:
- Use `great_expectations`, `pandera`, or custom Pandas/Polars assertions.
- Produce HTML/PDF reports with histograms of missingness, outlier rates, and station coverage heatmaps.
- Quantify impact of each cleaning rule on final sample size.

---

## 3. Data Validation Method

**Schema-level + semantic validation** (run after every pipeline stage):

- **Schema validation** (Pandera or Pydantic):
  - Required columns and dtypes.
  - Station ID exists in static stations table.
  - Timestamp monotonicity within station.
  - Capacity ≥ dock_bikes + free_bases (allow small tolerance for booking).

- **Business-rule validation**:
  - No trip with identical origin and destination and near-zero duration (except possible short loops).
  - Occupancy never exceeds capacity.
  - Net flow over a day is consistent with observed status changes (within measurement error).

- **Temporal validation**:
  - No future information in feature windows.
  - Train/val/test periods have comparable station coverage.

Fail the pipeline (or quarantine bad batches) if critical rules are violated. Log all violations with examples.

---

## 4. Statistical Profiling Method

Generate a comprehensive statistical profile of both raw and cleaned data.

**Core profiles**:

- **Univariate**: mean, median, std, min/max, quantiles (1%, 5%, 95%, 99%), skewness, kurtosis for travel_time, hourly starts/ends, occupancy ratio, net flow.
- **Temporal**: hourly, daily, weekly, monthly seasonality (boxplots + STL decomposition); autocorrelation (ACF/PACF) of station occupancy and demand.
- **Spatial**: station popularity ranking (starts + ends), Gini coefficient of demand concentration, average distance of trips.
- **Bivariate**: correlation matrix (occupancy vs. lagged demand, weather if available); contingency tables of user_type × ageRange.
- **System-level**: daily total trips, system-wide utilization proxy, frequency of empty (`dock_bikes = 0`) and full (`free_bases = 0`) events.

**Tools**: Pandas Profiling / ydata-profiling, Sweetviz, or custom Plotly + SciPy reports. Store summary tables and key plots in the thesis appendix.

---

## 5. Exploratory Data Analysis (EDA) Process

Follow a structured sequence that directly feeds feature engineering and modeling decisions.

1. **System overview**
   - Total trips, active stations evolution, growth trends 2017–2023.
   - Baseline bike utilization rate (define proxy: e.g., fraction of time bikes are away from stations, or 1 − average empty/full penalty).

2. **Temporal patterns**
   - Heatmaps of starts/ends by hour-of-day × day-of-week.
   - Seasonal and holiday effects.
   - Identification of peak vs. off-peak regimes.

3. **Spatial & imbalance patterns**
   - Map of average occupancy and empty/full frequency.
   - Stations that systematically empty in the morning and fill in the evening (or vice-versa).
   - Origin-destination flow matrix (top-N flows).

4. **Station clustering**
   - Cluster stations by demand profile (k-means or hierarchical on hourly average starts/ends + occupancy).
   - Useful later for hierarchical models or grouping constraints in the LP.

5. **Target exploration**
   - Distribution of net demand, ideal stock levels (e.g., capacity that minimizes empty + full risk in next horizon).
   - Correlation between current occupancy and future empty/full probability.

6. **Data issues discovery**
   - Confirm findings from quality assessment; quantify impact of cleaning.

Produce a short “EDA Insights” section that justifies every major modeling choice.

---

## 6. Features to Be Engineered

**Main modeling unit**: one row per `(station_id, time_bin)`. Target examples:
- Regression: expected net demand next *h* bins, or optimal target stock level.
- Classification: will the station be empty / full / critical in the next 30–60 min.

**Feature groups** (all must be causally available at prediction time):

### Temporal
- Hour, day-of-week, month, is_weekend, is_holiday (cyclical encoding: sin/cos).
- Time-bin index within day.

### Lag & rolling (station-specific)
- Starts, ends, net_flow, occupancy at lags 1, 2, 3, 4, 8, 12, 24, 48 bins.
- Rolling mean/std of starts/ends/occupancy over last 1 h, 3 h, 6 h, 24 h.
- Exponentially weighted moving averages.

### Station static / semi-static
- Capacity (`total_bases`).
- Historical average occupancy and demand by hour-of-week.
- Geographic coordinates (or embedded via clustering / graph embeddings).
- District / neighborhood indicators.
- Degree / centrality in the historical trip graph.

### Network / contextual
- Aggregated demand of nearest *k* stations (or stations within radius *r*).
- System-wide total bikes in circulation or average occupancy.
- Recent imbalance propagation (e.g., average net flow of neighbors).

### External (highly recommended)
- Weather: temperature, precipitation, wind (from AEMET or open Madrid air-quality stations), lagged and forecast if available.
- Special events / calendar (strikes, festivals, football matches) as binary or count features.

### Target engineering
- Continuous: future net demand, or a utilization-oriented target (e.g., stock level that historically maximized time-with-bike-available while keeping free docks).
- Classification: binary empty/full indicators or multi-class criticality levels at future horizons.

Store feature definitions in a configuration file so ablation studies are easy.

---

## 7. Machine Learning Models Design

**Problem framing**
- Primary: multi-horizon regression (predict continuous quantities that feed the utilization objective).
- Secondary: classification of critical states (empty/full) for operational alerts.
- Optional multi-task learning (shared backbone predicting both).

**Model families** (explicitly required by the thesis framing)

### 1. Random Forest
- Strong, interpretable tabular baseline.
- Use scikit-learn or cuML.
- Hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
- Feature importance (MDI + permutation) for thesis discussion.

### 2. XGBoost
- Expected strongest pure tabular performer.
- Native handling of missing values, early stopping, monotonic constraints if desired (e.g., higher past demand → higher predicted demand).
- Tune with Optuna or Hyperopt (`learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`/`lambda`).
- Support for multi-output or separate models per horizon.

### 3. Neural Network
- MLP on flattened tabular features (baseline NN).
- Prefer temporal models: LSTM / GRU / Temporal Convolutional Network on sequences of recent station states, optionally with station embeddings.
- Or a hybrid: station embedding + time features → dense layers, or a simple Graph Neural Network if modeling inter-station influence.
- Fit with scikit-learn's MLP. Use early stopping.

**Training & evaluation protocol**
- Time-series cross-validation (expanding or sliding window).
- Metrics:
  - Regression: MAE, RMSE, weighted MAE (higher weight on high-demand periods).
  - Classification: Precision/Recall/F1 for empty and full classes, PR-AUC.
  - Downstream: simulated utilization rate under predicted target stocks (with and without perfect redistribution).
- Ablations: remove feature groups, compare model families, vary horizon.
- Calibration of probabilistic outputs if uncertainty is used in the LP layer.

**Inference design**
- Batch scoring of the next 24 h for all stations.
- Output: recommended stock level (or range) per station per time bin → input to the linear program.

**Optional Linear Programming layer** (for the redistribution extension)
- Decision variables: bikes moved from *i* to *j* in time window *t* by vehicle *k*.
- Objective: maximize expected utilization (or minimize expected empty + full penalties + movement cost).
- Constraints: vehicle capacity, time windows, inventory balance (predicted demand/returns + moves), station capacity limits.
- Solver: PuLP, OR-Tools, or Gurobi/CPLEX (academic license).

---

## Recommended Implementation Order & Tools

1. Build and validate the data pipeline on one year → scale to full history.
2. Complete quality, validation, profiling, and EDA → freeze cleaning rules.
3. Engineer features and create the modeling table.
4. Train and compare RF → XGBoost → NN.
5. Evaluate end-to-end impact on utilization rate.
6. (Optional) Implement and test the LP redistribution planner.
7. Document pipeline performance (runtime, memory) and reproducibility (Docker + requirements + random seeds).

**Core stack**: Python 3.10+, Polars or Pandas, PyArrow, scikit-learn, XGBoost, Optuna, Plotly/Matplotlib/Seaborn, Pandera or Great Expectations, optionally Prefect or simple Makefiles for orchestration.

---

This guide is deliberately concrete so you can start coding immediately while keeping every required concept (pipeline design, quality, validation, statistical profiling, EDA, feature engineering, RF/XGBoost/NN, regression + classification, utilization rate) visible and interconnected.
