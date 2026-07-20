import pandas as pd
import joblib

# 1. Load the saved model package
model_package = joblib.load("models/yieldguard_model.pkl")

# 2. Take out the trained model and feature list
model = model_package["model"]
features = model_package["features"]

# 3. Create one new wafer lot
new_lot = pd.DataFrame([{
    "chamber_temp": 92.0,
    "chamber_pressure": 2.75,
    "etch_time": 66.0,
    "film_thickness": 112.0,
    "defect_count": 28
}])

# 4. Predict normal or high risk
prediction = model.predict(new_lot[features])[0]

# 5. Predict probability for each class
probabilities = model.predict_proba(new_lot[features])[0]

# Find where class 1 is located
class_list = list(model.classes_)
high_risk_position = class_list.index(1)

# Probability of class 1
risk_score = probabilities[high_risk_position]

# 6. Convert risk score into warning level
if risk_score >= 0.70:
    warning_level = "HIGH RISK"
elif risk_score >= 0.40:
    warning_level = "MEDIUM RISK"
else:
    warning_level = "LOW RISK"

# 7. Display input data
print("===== New Wafer Lot Input =====")
print(new_lot)

# 8. Display prediction result
print("\n===== YieldGuard AI Prediction =====")
print("Predicted label:", prediction)
print("Risk score:", round(risk_score * 100, 2), "%")
print("Warning level:", warning_level)

if prediction == 1:
    print("Meaning: This wafer lot is predicted to be high risk.")
else:
    print("Meaning: This wafer lot is predicted to be normal.")