import os
import pandas as pd
import streamlit as st

# 1. Configure the dashboard page
st.set_page_config(
    page_title="YieldGuard AI",
    page_icon="🧠",
    layout="wide"
)

# 2. Set prediction result file path
data_path = "outputs/explained_predictions.csv"

# 3. Dashboard title
st.title("YieldGuard AI")
st.caption(
    "Explainable wafer-lot yield-risk prediction and "
    "rule-based process risk analysis"
)

# 4. Check whether the result file exists
if not os.path.exists(data_path):
    st.error(
        "Prediction result file not found. "
        "Please run predict_batch.py and explain_risk.py first."
    )
    st.stop()

# 5. Read explained prediction results
df = pd.read_csv(data_path)

# 6. Check required dashboard columns
required_columns = [
    "risk_score_percent",
    "warning_level",
    "root_cause_summary",
    "recommended_action"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()

# 7. Calculate dashboard summary values
total_lots = len(df)

high_risk_count = (
    df["warning_level"] == "HIGH RISK"
).sum()

medium_risk_count = (
    df["warning_level"] == "MEDIUM RISK"
).sum()

low_risk_count = (
    df["warning_level"] == "LOW RISK"
).sum()

average_risk_score = df["risk_score_percent"].mean()

# 8. Display summary metrics
st.subheader("Production Risk Overview")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Total Wafer Lots",
    total_lots
)

metric_2.metric(
    "High Risk",
    high_risk_count
)

metric_3.metric(
    "Medium Risk",
    medium_risk_count
)

metric_4.metric(
    "Average Risk Score",
    f"{average_risk_score:.1f}%"
)

# 9. Display warning-level distribution
st.subheader("Warning Level Distribution")

risk_order = [
    "HIGH RISK",
    "MEDIUM RISK",
    "LOW RISK"
]

risk_distribution = (
    df["warning_level"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

st.bar_chart(risk_distribution)

# 10. Create sidebar filters
st.sidebar.header("Filters")

selected_warning_levels = st.sidebar.multiselect(
    "Warning Level",
    options=risk_order,
    default=risk_order
)

minimum_risk_score = st.sidebar.slider(
    "Minimum Risk Score",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)

# 11. Apply filters
filtered_df = df[
    df["warning_level"].isin(selected_warning_levels)
    & (
        df["risk_score_percent"]
        >= minimum_risk_score
    )
].copy()

filtered_df = filtered_df.sort_values(
    by="risk_score_percent",
    ascending=False
)

# 12. Select columns for the result table
display_columns = []

if "lot_id" in filtered_df.columns:
    display_columns.append("lot_id")

display_columns += [
    "risk_score_percent",
    "warning_level",
    "root_cause_summary",
    "recommended_action"
]

# 13. Display filtered prediction results
st.subheader("Wafer Lot Risk Results")

st.write(
    f"Showing {len(filtered_df)} of "
    f"{len(df)} wafer lots"
)

st.dataframe(
    filtered_df[display_columns],
    width="stretch",
    hide_index=True
)

# 14. Allow users to download filtered results
download_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Results",
    data=download_data,
    file_name="yieldguard_filtered_results.csv",
    mime="text/csv"
)

# 15. Display one selected wafer lot in detail
st.subheader("Inspect One Wafer Lot")

if "lot_id" in df.columns:
    lot_options = (
        df.sort_values(
            by="risk_score_percent",
            ascending=False
        )["lot_id"]
        .astype(str)
        .tolist()
    )

    selected_lot = st.selectbox(
        "Select Wafer Lot",
        options=lot_options
    )

    selected_row = df[
        df["lot_id"].astype(str) == selected_lot
    ].iloc[0]

    detail_1, detail_2, detail_3 = st.columns(3)

    detail_1.metric(
        "Risk Score",
        f"{selected_row['risk_score_percent']:.1f}%"
    )

    detail_2.metric(
        "Warning Level",
        selected_row["warning_level"]
    )

    if "predicted_risk_label" in selected_row.index:
        detail_3.metric(
            "Predicted Label",
            int(selected_row["predicted_risk_label"])
        )

    st.markdown("#### Process Conditions")

    process_columns = [
        "chamber_temp",
        "chamber_pressure",
        "etch_time",
        "film_thickness",
        "defect_count"
    ]

    available_process_columns = [
        column for column in process_columns
        if column in selected_row.index
    ]

    process_details = pd.DataFrame({
        "Feature": available_process_columns,
        "Value": [
            selected_row[column]
            for column in available_process_columns
        ]
    })

    st.dataframe(
        process_details,
        width="stretch",
        hide_index=True
    )

    st.markdown("#### Possible Risk Factors")

    st.warning(
        selected_row["root_cause_summary"]
    )

    st.markdown("#### Recommended Action")

    st.info(
        selected_row["recommended_action"]
    )

else:
    st.info(
        "No lot_id column is available for "
        "individual wafer-lot inspection."
    )

# 16. Add MVP disclaimer
st.divider()

st.caption(
    "MVP disclaimer: This dashboard currently uses synthetic "
    "semiconductor-style data and rule-based explanations. "
    "The identified factors are possible risk indicators, "
    "not confirmed physical root causes."
)