import pandas as pd
import os

# 1. Read batch prediction results
input_path = "outputs/batch_predictions.csv"
df = pd.read_csv(input_path)

# 2. Check required columns
required_columns = [
    "chamber_temp",
    "chamber_pressure",
    "etch_time",
    "film_thickness",
    "defect_count",
    "risk_score_percent",
    "warning_level"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# 3. Analyse possible process risk factors
def analyse_root_causes(row):
    causes = []
    actions = []

    if row["defect_count"] >= 15:
        causes.append("High defect count")
        actions.append("Review inspection results and defect source")

    if row["chamber_temp"] > 90:
        causes.append("Chamber temperature too high")
        actions.append("Check chamber temperature control")

    if row["chamber_pressure"] > 2.6:
        causes.append("Chamber pressure too high")
        actions.append("Inspect pressure control and vacuum system")

    if row["etch_time"] > 65:
        causes.append("Etch time too long")
        actions.append("Review etch recipe and process duration")

    if row["film_thickness"] > 110:
        causes.append("Film thickness too high")
        actions.append("Check deposition rate and recipe settings")

    elif row["film_thickness"] < 90:
        causes.append("Film thickness too low")
        actions.append("Check deposition uniformity and process settings")

    if not causes:
        causes.append(
            "No single rule-based deviation identified"
        )
        actions.append(
            "Review combined process conditions and model risk score"
        )

    return pd.Series({
        "root_cause_summary": "; ".join(causes),
        "recommended_action": "; ".join(actions)
    })


# 4. Apply root-cause analysis to every wafer lot
explanations = df.apply(
    analyse_root_causes,
    axis=1
)

# 5. Add explanations to results
results = pd.concat(
    [df, explanations],
    axis=1
)

# 6. Create output folder
os.makedirs("outputs", exist_ok=True)

# 7. Save explained results
output_path = "outputs/explained_predictions.csv"

results.to_csv(
    output_path,
    index=False
)

# 8. Select columns to display
display_columns = []

if "lot_id" in results.columns:
    display_columns.append("lot_id")

display_columns += [
    "risk_score_percent",
    "warning_level",
    "root_cause_summary",
    "recommended_action"
]

# 9. Display high-risk wafer lots
high_risk_results = results[
    results["warning_level"] == "HIGH RISK"
]

print("===== High-Risk Process Analysis =====")
print(high_risk_results[display_columns].head(10).to_string(index=False))

# 10. Display summary
print("\n===== Process Analysis Summary =====")
print("Total wafer lots:", len(results))
print("High-risk wafer lots:", len(high_risk_results))

print("\n===== File Saved =====")
print("Root-cause analysis completed successfully!")
print("Saved location:", output_path)