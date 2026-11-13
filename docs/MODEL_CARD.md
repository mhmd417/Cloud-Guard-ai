# Model Card: CloudGuard AI Predictive Maintenance Classifier (`v1.0.0`)

## 1. Model Overview
* **Model Name:** CloudGuard AI Predictive Maintenance Classifier
* **Model Version:** `1.0.0`
* **Artifact Path:** `ml/models/cloudguard_rf_v1.0.0.joblib`
* **Metadata Path:** `ml/models/cloudguard_rf_v1.0.0_metadata.json`
* **Architecture:** Scikit-learn `Pipeline` combining a `ColumnTransformer` (`OneHotEncoder` on `Type` + `StandardScaler` on 5 numeric sensor readings) and a `RandomForestClassifier` (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`, `class_weight="balanced"`, `random_state=42`).
* **Decision Threshold:** `0.45` (predicts failure class `1` when predicted failure probability >= `0.45`, otherwise `0`).

---

## 2. Intended Use
* **Primary Purpose:** Educational and portfolio demonstration showing how a trained scikit-learn pipeline serves real-time machine failure risk predictions through a FastAPI backend and web interface.
* **Intended Users:** Students, software engineers, and ML learners studying end-to-end ML engineering, API design, and containerization.
* **Out-of-Scope / Prohibited Use:** **NOT validated for real industrial deployment, safety-critical alerting, or automated equipment shutdown.** Do not use this model to make maintenance decisions on physical machinery.

---

## 3. Data Source & Feature Specification
* **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (`ai4i2020.csv`, 10,000 rows total: 8,000 stratified training rows, 2,000 stratified test rows).
* **Synthetic Data Disclosure:** **This dataset is 100% synthetic.** It was generated algorithmically to simulate milling machine telemetry where failures follow deterministic mathematical rules (heat dissipation, power limits, overstrain, and tool wear). High benchmark scores on this dataset reflect learning those synthetic rules and **do not prove real-world industrial performance**.
* **No Advance-Warning Window:** Each row represents an instantaneous snapshot of machine operating conditions paired with a binary failure label (`Machine failure`). The dataset contains no timestamps or time-series lead windows; therefore, **this model does not predict how many minutes or hours in advance a failure will occur**.

### Input Features (6 total)
1. `Type` (categorical: `L`, `M`, or `H` product quality variant)
2. `Air temperature [K]` (numeric, Kelvin)
3. `Process temperature [K]` (numeric, Kelvin)
4. `Rotational speed [rpm]` (numeric, revolutions per minute)
5. `Torque [Nm]` (numeric, Newton-meters)
6. `Tool wear [min]` (numeric, cumulative tool wear in minutes)

### Excluded Columns (Leakage Prevention)
* **Identifiers:** `UDI`, `Product ID`
* **Failure Sub-type Indicators (direct target leakage):** `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failures)

---

## 4. Evaluation Metrics (Held-Out Test Set)
Because machine failures are rare (~3.4% positive class), accuracy alone is misleading (a dummy model predicting "no failure" always achieves ~96.6% accuracy while catching 0% of failures). Model selection and evaluation prioritize **PR-AUC**, **Recall**, **Precision**, and **F1-Score**.

Evaluated **once** on the held-out test set (n = 2,000, 68 true failures) at decision threshold `0.45`:

| Metric | Test Score | Interpretation |
| :--- | :--- | :--- |
| **PR-AUC** | **0.7253** | Primary ranking metric under severe class imbalance |
| **ROC-AUC** | **0.9658** | Overall class separation across all thresholds |
| **Recall** | **0.8235** | Caught **56 of 68** actual failures (82.35%) |
| **Precision** | **0.4912** | 56 true alarms out of 114 total failure alerts |
| **F1-Score** | **0.6154** | Harmonic mean of Precision and Recall |

### Test Confusion Matrix (`threshold = 0.45`)
* **True Negatives (TN):** `1,874` (normal operation correctly classified as normal)
* **False Positives (FP):** `58` (false alarms on normal operation)
* **False Negatives (FN):** `12` (missed failures)
* **True Positives (TP):** `56` (actual failures detected)

### Inference Latency (Single-Row API Simulation)
* **Mean Latency:** `34.09 ms` per prediction (measured over 200 sequential single-row calls)
* **95th Percentile (p95) Latency:** `41.44 ms` per prediction

---

## 5. Key Limitations & Risks
1. **Synthetic Rule Memorization:** The model learns synthetic boundaries specific to the AI4I 2020 generator and will not generalize to real factory sensors without retraining on real telemetry.
2. **No Temporal Context:** Predictions evaluate single static readings in isolation and cannot detect gradual sensor drift or time-to-failure horizons.
3. **False Alarm Trade-Off:** At threshold `0.45`, roughly half of positive alerts are false alarms (`58 FP` vs. `56 TP`) in order to catch `82.35%` of failures (`12 FN`).
4. **Out-of-Range Inputs:** Extreme sensor values far outside the training distribution (`Air temperature ~295–305 K`, `Rotational speed ~1168–2886 rpm`, `Torque ~3.8–76.6 Nm`) may produce unreliable probability estimates.
