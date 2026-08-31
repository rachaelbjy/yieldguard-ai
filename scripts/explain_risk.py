import os
import pandas as pd


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
    "warning_level",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# 3. Analyse possible process risk factors
def analyse_risk_factors(row):
    factors = []
    actions = []

    if row["defect_count"] >= 15:
        factors.append(
            "High defect count"
        )
        actions.append(
            "Review inspection results and defect source"
        )

    if row["chamber_temp"] > 90:
        factors.append(
            "Chamber temperature too high"
        )
        actions.append(
            "Check chamber temperature control"
        )

    if row["chamber_pressure"] > 2.6:
        factors.append(
            "Chamber pressure too high"
        )
        actions.append(
            "Inspect pressure control and vacuum system"
        )

    if row["etch_time"] > 65:
        factors.append(
            "Etch time too long"
        )
        actions.append(
            "Review etch recipe and process duration"
        )

    if row["film_thickness"] > 110:
        factors.append(
            "Film thickness too high"
        )
        actions.append(
            "Check deposition rate and recipe settings"
        )

    elif row["film_thickness"] < 90:
        factors.append(
            "Film thickness too low"
        )
        actions.append(
            "Check deposition uniformity and process settings"
        )

    if not factors:
        factors.append(
            "No single rule-based deviation identified"
        )
        actions.append(
            "Review combined process conditions and model risk score"
        )

    return pd.Series({
        "root_cause_summary": "; ".join(factors),
        "recommended_action": "; ".join(actions),
    })


# 4. Apply process-risk analysis to every wafer lot
explanations = df.apply(
    analyse_risk_factors,
    axis=1,
)


# 5. Add explanations to results
results = pd.concat(
    [df, explanations],
    axis=1,
)


# 6. Create output folder
os.makedirs(
    "outputs",
    exist_ok=True,
)


# 7. Save explained results
output_path = "outputs/explained_predictions.csv"

results.to_csv(
    output_path,
    index=False,
)


# 8. Select high-risk wafer lots
high_risk_results = results[
    results["warning_level"] == "HIGH RISK"
]


# 9. Display high-risk wafer lots
print(
    "===== High-Risk Process Analysis ====="
)

for number, (_, row) in enumerate(
    high_risk_results.head(10).iterrows(),
    start=1,
):
    print()

    if "lot_id" in row.index:
        print(
            f"{number}. Wafer Lot: {row['lot_id']}"
        )
    else:
        print(
            f"{number}. Wafer Lot"
        )

    print(
        f"   Risk Score: {row['risk_score_percent']}% "
        f"| Warning Level: {row['warning_level']}"
    )

    print(
        "   Possible Risk Factors:"
    )

    for factor in str(
        row["root_cause_summary"]
    ).split(";"):
        print(
            f"   - {factor.strip()}"
        )

    print(
        "   Recommended Actions:"
    )

    for action in str(
        row["recommended_action"]
    ).split(";"):
        print(
            f"   - {action.strip()}"
        )


# 10. Display summary
print(
    "\n===== Process Analysis Summary ====="
)
print(
    "Total wafer lots:",
    len(results),
)
print(
    "High-risk wafer lots:",
    len(high_risk_results),
)

print(
    "\n===== File Saved ====="
)
print(
    "Process analysis completed successfully!"
)
print(
    "Saved location:",
    output_path,
)