import pandas as pd

# Read the wafer dataset
df = pd.read_csv("data/wafer_data.csv")

print("===== 1. First 5 rows =====")
print(df.head())

print("\n===== 2. Dataset shape =====")
print(df.shape)

print("\n===== 3. Column names =====")
print(df.columns.tolist())

print("\n===== 4. Missing values =====")
print(df.isnull().sum())

print("\n===== 5. Risk label count =====")
print(df["risk_label"].value_counts())

print("\n===== 6. Basic statistics =====")
print(df.describe())

print("\n===== 7. Average yield by risk label =====")
print(df.groupby("risk_label")["final_yield"].mean())