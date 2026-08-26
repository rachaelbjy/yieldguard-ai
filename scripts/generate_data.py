import pandas as pd
import random
import os

random.seed(42)

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Store all wafer lot records here
records = []

# Create 200 fake wafer lots
for i in range(1, 201):
    lot_id = f"LOT_{i:03d}"

    tool_id = random.choice(["TOOL_A", "TOOL_B", "TOOL_C", "TOOL_D"])

    chamber_temp = random.normalvariate(85, 4)
    chamber_pressure = random.normalvariate(2.2, 0.25)
    etch_time = random.normalvariate(60, 4)
    film_thickness = random.normalvariate(100, 6)

    # Basic defect logic:
    # If process values drift away from normal, defect count increases.
    defect_count = random.randint(3, 15)

    if chamber_temp > 90:
        defect_count += random.randint(5, 15)

    if chamber_pressure > 2.6:
        defect_count += random.randint(5, 15)

    if etch_time > 65:
        defect_count += random.randint(3, 10)

    if film_thickness > 110 or film_thickness < 90:
        defect_count += random.randint(5, 15)

    # Higher defect count usually means lower yield
    final_yield = 96 - (defect_count * 0.8) + random.normalvariate(0, 2)

    # Keep yield between 50 and 99
    final_yield = max(50, min(99, final_yield))

    # Define high risk
    risk_label = 1 if final_yield < 85 else 0

    records.append({
        "lot_id": lot_id,
        "tool_id": tool_id,
        "chamber_temp": round(chamber_temp, 2),
        "chamber_pressure": round(chamber_pressure, 2),
        "etch_time": round(etch_time, 2),
        "film_thickness": round(film_thickness, 2),
        "defect_count": defect_count,
        "final_yield": round(final_yield, 2),
        "risk_label": risk_label
    })

# Convert records into table
df = pd.DataFrame(records)

# Save as CSV
df.to_csv("data/wafer_data.csv", index=False)

print("Synthetic wafer dataset created successfully!")
print(df.head())
print()
print("Dataset shape:", df.shape)
print("High-risk lots:", df["risk_label"].sum())