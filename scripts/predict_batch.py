import pandas as pd
import joblib
import os

# 1. Load the saved model package
model_package = joblib.load("models/yieldguard_model.pkl")

model = model_package["model"]
features = model_package["features"]

# 2. Read the batch wafer-lot data
input_path = "data/wafer_data.csv"
df = pd.read_csv(input_path)

# 3. Check whether all required features exist
missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required columns: {missing_features}"
    )

# 4. Select model input features
X_batch = df[features]

# 5. Predict risk labels for all wafer lots
predictions = model.predict(X_batch)

# 6. Predict probabilities for all wafer lots
all_probabilities = model.predict_proba(X_batch)

class_list = list(model.classes_)
high_risk_position = class_list.index(1)

risk_scores = all_probabilities[:, high_risk_position]

# 7. Create a copy for prediction results
results = df.copy()

results["predicted_risk_label"] = predictions
results["risk_score"] = risk_scores

# 8. Convert risk score into warning level
def get_warning_level(score):
    if score >= 0.70:
        return "HIGH RISK"
    elif score >= 0.40:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


results["warning_level"] = results["risk_score"].apply(
    get_warning_level
)

# 9. Convert risk score into percentage
results["risk_score_percent"] = (
    results["risk_score"] * 100
).round(2)

# 10. Create output folder
os.makedirs("outputs", exist_ok=True)

# 11. Save prediction results
output_path = "outputs/batch_predictions.csv"

results.to_csv(
    output_path,
    index=False
)

# 12. Choose columns to display
display_columns = []

if "lot_id" in results.columns:
    display_columns.append("lot_id")

display_columns += features

display_columns += [
    "predicted_risk_label",
    "risk_score_percent",
    "warning_level"
]

# 13. Display the first 10 prediction results
print("===== Batch Prediction Results =====")
print(results[display_columns].head(10))

# 14. Display warning-level summary
print("\n===== Warning Level Summary =====")
print(results["warning_level"].value_counts())

# 15. Display output location
print("\n===== File Saved =====")
print("Batch prediction completed successfully!")
print("Saved location:", output_path)