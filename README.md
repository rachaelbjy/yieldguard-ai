# YieldGuard AI

YieldGuard AI is a machine-learning decision-support prototype for predicting wafer-lot yield risk in semiconductor manufacturing.

It was developed as an MVP for the **MAIC Nexus Challenge**. The project uses a Random Forest model to estimate wafer-lot risk, SHAP to explain individual predictions, and simple rule-based checks to highlight process conditions that may need further attention.

The current version uses synthetic semiconductor-style data because real fab production data is usually confidential.

## What It Does

YieldGuard AI takes wafer-lot process data and produces:

- A predicted risk class
- A high-risk probability score
- LOW / MEDIUM / HIGH warning levels
- SHAP explanations for individual predictions
- Possible process risk factors
- Suggested follow-up actions
- An interactive Streamlit dashboard

The goal is to help users identify higher-risk wafer lots and understand which inputs influenced the model prediction.

The system is designed as a decision-support tool, not as a replacement for process or yield engineers.

## Model and Data

The current model is a **Random Forest classifier** built with scikit-learn.

It uses five input features:

| Feature | Description |
| --- | --- |
| `chamber_temp` | Chamber temperature |
| `chamber_pressure` | Chamber pressure |
| `etch_time` | Etching process duration |
| `film_thickness` | Film thickness measurement |
| `defect_count` | Number of detected defects |

`lot_id` is used to identify wafer lots but is not used as a model feature.

`tool_id` is included in the synthetic dataset but is not currently used by the model.

`final_yield` is also excluded from the model because `risk_label` is derived from it. Including `final_yield` would cause target leakage.

The synthetic dataset contains 200 wafer lots and is generated using a fixed random seed so that the same data can be reproduced.

The data is split into:

```text
80% training data
20% held-out test data
```

The held-out rows are saved as:

```text
data/unseen_test_data.csv
```

This file can be uploaded to the dashboard to evaluate the model on data that was not used during training.

## Risk Levels

The Random Forest predicts a binary class:

```text
0 = Low Risk
1 = High Risk
```

The model also produces a probability for the high-risk class.

The dashboard converts this probability into three warning levels:

| Risk Score | Warning Level |
| --- | --- |
| >= 70% | HIGH RISK |
| 40% to < 70% | MEDIUM RISK |
| < 40% | LOW RISK |

The binary model prediction and the dashboard warning level use different decision thresholds, so a lot can have a predicted class of `1` while still being shown as `MEDIUM RISK`.

These warning thresholds are part of the MVP and are not universal semiconductor process limits.

## Explainability

YieldGuard AI uses **SHAP** to explain individual Random Forest predictions.

For a selected wafer lot, the dashboard shows:

- The value of each model feature
- Its SHAP contribution
- Whether it pushes the prediction toward HIGH RISK or LOW RISK
- The feature with the strongest influence on the prediction
- A SHAP contribution chart

SHAP explains how the model arrived at a prediction. It does not prove that a feature physically caused a defect or yield loss.

The project also includes simple rule-based process checks. These look for conditions such as:

- High defect count
- High chamber temperature
- High chamber pressure
- Long etch time
- High or low film thickness

The rules generate possible risk indicators and suggested follow-up actions.

They are illustrative MVP rules and should not be treated as confirmed semiconductor root causes or real fab process specifications.

## Dashboard

The Streamlit dashboard includes:

- Built-in synthetic demo data
- CSV upload
- Live model prediction
- Input validation
- Production risk overview
- LOW / MEDIUM / HIGH distribution
- Risk filters
- Downloadable prediction results
- Individual wafer-lot inspection
- SHAP explanation
- Possible risk factors
- Recommended actions

If an uploaded CSV also contains a valid `risk_label` column, the dashboard can display:

- Accuracy
- Precision
- Recall
- Confusion Matrix

For meaningful evaluation, the labelled data should be held out from model training.

## How It Works

```text
Wafer-Lot Data
      |
      v
Input Validation
      |
      v
Random Forest Model
      |
      +----------------------+
      |                      |
      v                      v
Predicted Class       High-Risk Probability
                             |
                             v
                     Warning Level
                             |
              +--------------+--------------+
              |                             |
              v                             v
       SHAP Explanation            Rule-Based Checks
              |                             |
              v                             v
       Model Drivers              Possible Risk Factors
                                  Recommended Actions
              |                             |
              +--------------+--------------+
                             |
                             v
                    Streamlit Dashboard
```

## Project Structure

```text
yieldguard-ai/
|
├── app.py
├── README.md
├── requirements.txt
|
├── data/
│   ├── wafer_data.csv
│   └── unseen_test_data.csv
|
├── models/
│   └── yieldguard_model.pkl
|
├── outputs/
│   ├── batch_predictions.csv
│   └── explained_predictions.csv
|
└── scripts/
    ├── generate_data.py
    ├── check_data.py
    ├── train_model.py
    ├── predict_one_lot.py
    ├── predict_batch.py
    └── explain_risk.py
```

## Running the Project

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Generate the synthetic dataset:

```bash
python scripts/generate_data.py
```

Check the generated data:

```bash
python scripts/check_data.py
```

Train the Random Forest model:

```bash
python scripts/train_model.py
```

Generate batch prediction results:

```bash
python scripts/predict_batch.py
```

Generate rule-based process analysis:

```bash
python scripts/explain_risk.py
```

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The dashboard will normally open at:

```text
http://localhost:8501
```

The dashboard can also perform predictions directly from an uploaded CSV, so the batch prediction scripts are mainly useful for generating saved output files and checking the pipeline separately.

## CSV Input

A CSV uploaded to the dashboard must contain the five model features.

Example:

```csv
lot_id,chamber_temp,chamber_pressure,etch_time,film_thickness,defect_count
LOT_001,85.10,2.05,54.92,99.54,15
LOT_002,78.36,1.87,54.54,106.51,5
LOT_003,93.88,2.16,60.45,89.55,31
```

Required columns:

```text
chamber_temp
chamber_pressure
etch_time
film_thickness
defect_count
```

`lot_id` is optional for prediction, but it is recommended because it enables individual wafer-lot inspection in the dashboard.

For model evaluation, a labelled CSV may also contain:

```text
risk_label
```

where the values must be `0` or `1`.

## Current Limitations

This project is an MVP and has several limitations:

- The dataset is synthetic rather than real fab production data.
- Only five model features are currently used.
- `tool_id` is not included in model training.
- `defect_count` has a strong relationship with yield because of the synthetic data-generation logic.
- The process thresholds are illustrative rather than validated fab specifications.
- SHAP explains model behaviour but does not establish physical causality.
- The model has not been validated in a real semiconductor manufacturing environment.

A real implementation would require actual manufacturing data, process-specific engineering validation, appropriate feature timing, model monitoring, and integration with fab systems.

## Requirements

The project uses:

```text
pandas
numpy
scikit-learn
joblib
streamlit
shap
```

## Disclaimer

YieldGuard AI is an educational and competition MVP.

Its predictions, SHAP explanations, process indicators, thresholds, and recommended actions are intended for demonstration purposes using synthetic data.

The system should be treated as a decision-support prototype and not as a validated semiconductor production tool.