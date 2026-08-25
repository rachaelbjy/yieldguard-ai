import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report

# 1. Read dataset
df = pd.read_csv("data/wafer_data.csv")

# 2. Choose input features
# We do NOT use final_yield because final_yield is the result we only know after testing.
features = [
    "chamber_temp",
    "chamber_pressure",
    "etch_time",
    "film_thickness",
    "defect_count"
]

X = df[features]

# 3. Choose label / answer
y = df["risk_label"]

# 4. Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save unseen test data for later evaluation
test_data = df.loc[X_test.index].copy()

test_data.to_csv(
    "data/unseen_test_data.csv",
    index=False
)

# 5. Create AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Make predictions on test data
y_pred = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("===== Model Performance =====")
print("Accuracy:", round(accuracy, 3))
print("Recall:", round(recall, 3))
print("Precision:", round(precision, 3))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== Full Report =====")
print(classification_report(y_test, y_pred))

# 9. Show feature importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance)

# 10. Save trained model
os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "features": features
}

joblib.dump(model_package, "models/yieldguard_model.pkl")

print("\n===== Model Saved =====")
print("Model saved successfully!")
print("Saved location: models/yieldguard_model.pkl")