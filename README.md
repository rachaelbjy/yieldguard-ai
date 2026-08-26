# YieldGuard AI

**YieldGuard AI** is an explainable AI decision-support system for semiconductor wafer-lot yield-risk prediction.

Developed as an MVP for the **MAIC Nexus Challenge**, it combines machine learning, SHAP explainability, process-risk analysis, and an interactive dashboard to identify potentially high-risk wafer lots and provide interpretable insights for manufacturing decision support.

> **Note:** The current MVP uses synthetic semiconductor-style data and is intended for demonstration and development purposes.

---

## Problem

Semiconductor manufacturing involves tightly controlled process conditions. Deviations or combinations of abnormal process parameters may contribute to higher defect rates and lower wafer yield.

Manufacturing teams therefore need ways to identify potentially risky wafer lots early and understand which process factors may deserve further investigation.

However, semiconductor production data is often complex, high-dimensional, and difficult to interpret using prediction scores alone.

---

## Solution

YieldGuard AI analyses wafer-lot process data using a **Random Forest classifier** and generates:

- Predicted wafer-lot risk
- High-risk probability score
- LOW / MEDIUM / HIGH warning levels
- SHAP-based model explanations
- Possible process risk indicators
- Recommended engineering actions

The results are presented through an interactive **Streamlit dashboard**.

The goal is not to replace engineers, but to provide an interpretable AI-based decision-support layer that helps users identify which wafer lots may require closer attention.

---

## Key Features

### Machine Learning Prediction

The current model uses five input features:

- `chamber_temp`
- `chamber_pressure`
- `etch_time`
- `film_thickness`
- `defect_count`

A Random Forest classifier predicts whether a wafer lot belongs to the high-risk class.

The model also generates a high-risk probability that is converted into a risk score.

### Risk Warning System

Risk scores are converted into three dashboard warning levels:

- **HIGH RISK:** >= 70%
- **MEDIUM RISK:** >= 40% and < 70%
- **LOW RISK:** < 40%

These thresholds are MVP decision-support rules rather than universal semiconductor process standards.

### Explainable AI with SHAP

YieldGuard AI uses **SHAP** to explain individual Random Forest predictions.

For a selected wafer lot, the dashboard shows:

- Feature values
- SHAP contributions
- Contribution direction
- Strongest AI prediction driver
- SHAP contribution bar chart

Positive SHAP values push the model prediction toward **HIGH RISK**, while negative values push it toward **LOW RISK**.

### Process Risk Analysis

The system also applies simple engineering-inspired rules to identify possible process deviations, including:

- High defect count
- High chamber temperature
- High chamber pressure
- Long etch time
- High film thickness
- Low film thickness

Corresponding recommended actions are generated to support further investigation.

These outputs represent **possible risk indicators**, not confirmed physical root causes.

### Interactive Dashboard

The Streamlit dashboard supports:

- CSV upload
- Live model prediction
- Built-in demo data
- Production risk overview
- Risk filtering
- Warning-level distribution
- Individual wafer-lot inspection
- SHAP explanation
- Model evaluation
- Confusion Matrix
- Recommended actions
- Downloadable filtered results

---

## How It Works

```text
Wafer-Lot CSV
      ↓
Input Validation
      ↓
Model Features
      ↓
Random Forest
      ↓
Risk Prediction + Probability
      ↓
LOW / MEDIUM / HIGH Warning
      ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓
SHAP          Rule-Based
Explanation   Process Analysis
 ↓               ↓
AI Drivers    Risk Indicators
              + Actions
 └───────────────┴──────────────────┘
                 ↓
        Streamlit Dashboard
```

---

## Model Evaluation

The synthetic dataset is split into:

```text
80% Training Data
20% Held-Out Test Data
```

The model is trained only on the training portion.

The held-out test set is saved as:

```text
data/unseen_test_data.csv
```

When labelled test data is uploaded, the dashboard can calculate:

- Accuracy
- Precision
- Recall
- Confusion Matrix

This allows model performance to be evaluated on wafer lots that were not used during model fitting.

Model results should be interpreted as performance on the current **synthetic MVP dataset**, not as validated semiconductor-fab performance.

---

## Project Structure

```text
yieldguard-ai/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── wafer_data.csv
│   └── unseen_test_data.csv
│
├── models/
│   └── yieldguard_model.pkl
│
├── outputs/
│   ├── batch_predictions.csv
│   └── explained_predictions.csv
│
└── scripts/
    ├── generate_data.py
    ├── check_data.py
    ├── train_model.py
    ├── predict_one_lot.py
    ├── predict_batch.py
    └── explain_risk.py
```

---

## Installation

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Current dependencies:

```text
pandas
numpy
scikit-learn
joblib
streamlit
shap
```

---

## Running YieldGuard AI

### 1. Generate the synthetic dataset

```bash
python scripts/generate_data.py
```

### 2. Train the model

```bash
python scripts/train_model.py
```

### 3. Generate batch predictions

```bash
python scripts/predict_batch.py
```

### 4. Generate rule-based explanations

```bash
python scripts/explain_risk.py
```

### 5. Start the dashboard

```bash
python -m streamlit run app.py
```

The dashboard will normally run at:

```text
http://localhost:8501
```

Users can then upload a CSV containing the required model features for live prediction.

---

## Example CSV Input

```csv
lot_id,chamber_temp,chamber_pressure,etch_time,film_thickness,defect_count
LOT_001,85.10,2.05,54.92,99.54,15
LOT_002,78.36,1.87,54.54,106.51,5
LOT_003,93.88,2.16,60.45,89.55,31
```

Required model columns:

```text
chamber_temp
chamber_pressure
etch_time
film_thickness
defect_count
```

`lot_id` is recommended for individual wafer-lot inspection.

---

## Current Limitations

YieldGuard AI is currently an MVP.

Key limitations include:

- The dataset is synthetic rather than real fab production data.
- Only five model features are currently used.
- `tool_id` is not yet included in model training.
- `defect_count` has strong predictive influence because of the synthetic data-generation logic.
- Process thresholds are illustrative assumptions rather than validated fab specifications.
- SHAP explains model behaviour but does not prove physical causality.
- The system has not been validated in a real semiconductor production environment.

Real deployment would require real manufacturing data, engineering validation, process-specific thresholds, model monitoring, and integration with existing manufacturing systems.

---

## Disclaimer

YieldGuard AI is an educational and competition MVP demonstrating the potential use of explainable machine learning in semiconductor manufacturing.

Model predictions, SHAP explanations, process-risk indicators, thresholds, and recommended actions are provided for demonstration purposes.

The system is intended as a **decision-support tool** and should not replace process engineers, yield engineers, equipment engineers, or other semiconductor domain experts.