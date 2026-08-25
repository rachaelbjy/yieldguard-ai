import os
import pandas as pd
import joblib
import streamlit as st


# 1. Configure dashboard
st.set_page_config(
    page_title="YieldGuard AI",
    page_icon="🧠",
    layout="wide"
)

st.title("YieldGuard AI")
st.caption(
    "Explainable wafer-lot yield-risk prediction and "
    "rule-based process risk analysis"
)


# 2. Set file paths
model_path = "models/yieldguard_model.pkl"
demo_data_path = "outputs/explained_predictions.csv"


# 3. Check model file
if not os.path.exists(model_path):
    st.error(
        "Trained model not found. "
        "Please run train_model.py first."
    )
    st.stop()


# 4. Load trained model
model_package = joblib.load(model_path)

model = model_package["model"]
features = model_package["features"]


# 5. Warning-level function
def get_warning_level(score):
    if score >= 0.70:
        return "HIGH RISK"
    elif score >= 0.40:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


# 6. Root-cause explanation function
def analyse_root_causes(row):
    causes = []
    actions = []

    if row["defect_count"] >= 15:
        causes.append("High defect count")
        actions.append(
            "Review inspection results and defect source"
        )

    if row["chamber_temp"] > 90:
        causes.append("Chamber temperature too high")
        actions.append(
            "Check chamber temperature control"
        )

    if row["chamber_pressure"] > 2.6:
        causes.append("Chamber pressure too high")
        actions.append(
            "Inspect pressure control and vacuum system"
        )

    if row["etch_time"] > 65:
        causes.append("Etch time too long")
        actions.append(
            "Review etch recipe and process duration"
        )

    if row["film_thickness"] > 110:
        causes.append("Film thickness too high")
        actions.append(
            "Check deposition rate and recipe settings"
        )

    elif row["film_thickness"] < 90:
        causes.append("Film thickness too low")
        actions.append(
            "Check deposition uniformity and process settings"
        )

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


# 7. CSV upload
st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload wafer-lot CSV",
    type=["csv"]
)


# 8. Use uploaded data or demo data
if uploaded_file is not None:

    input_df = pd.read_csv(uploaded_file)

    if input_df.empty:
        st.error("The uploaded CSV contains no wafer lots.")
        st.stop()

    # Check required model features
    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        st.error(
            f"Missing required columns: {missing_features}"
        )
        st.stop()

    # Convert required features to numeric values
    for feature in features:
        input_df[feature] = pd.to_numeric(
            input_df[feature],
            errors="coerce"
        )

    # Check for invalid or missing values
    if input_df[features].isnull().any().any():
        st.error(
            "Required feature columns contain missing "
            "or non-numeric values."
        )
        st.stop()

    # Prepare model input
    X_batch = input_df[features]

    # Predict labels
    predictions = model.predict(X_batch)

    # Predict probabilities
    all_probabilities = model.predict_proba(X_batch)

    class_list = list(model.classes_)
    high_risk_position = class_list.index(1)

    risk_scores = all_probabilities[
        :,
        high_risk_position
    ]

    # Create result table
    df = input_df.copy()

    df["predicted_risk_label"] = predictions
    df["risk_score"] = risk_scores

    df["risk_score_percent"] = (
        df["risk_score"] * 100
    ).round(2)

    df["warning_level"] = (
        df["risk_score"]
        .apply(get_warning_level)
    )

    # Generate explanations
    explanations = df.apply(
        analyse_root_causes,
        axis=1
    )

    df = pd.concat(
        [df, explanations],
        axis=1
    )

    st.success(
        f"Successfully analysed {len(df)} wafer lots."
    )

else:

    # Use existing demo results when no file is uploaded
    if not os.path.exists(demo_data_path):
        st.info(
            "Upload a wafer-lot CSV to begin prediction."
        )
        st.stop()

    df = pd.read_csv(demo_data_path)

    st.sidebar.caption(
        "Currently displaying the built-in demo dataset."
    )


# 9. Validate dashboard columns
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


# 10. Calculate summary values
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

average_risk_score = (
    df["risk_score_percent"].mean()
)


# 11. Display summary metrics
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


# 12. Warning-level distribution
st.subheader("Warning Level Distribution")

risk_order = [
    "HIGH RISK",
    "MEDIUM RISK",
    "LOW RISK"
]

risk_distribution = (
    df["warning_level"]
    .value_counts()
    .reindex(
        risk_order,
        fill_value=0
    )
)

st.bar_chart(risk_distribution)


# 13. Sidebar filters
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


# 14. Apply filters
filtered_df = df[
    df["warning_level"].isin(
        selected_warning_levels
    )
    & (
        df["risk_score_percent"]
        >= minimum_risk_score
    )
].copy()

filtered_df = filtered_df.sort_values(
    by="risk_score_percent",
    ascending=False
)


# 15. Select result-table columns
display_columns = []

if "lot_id" in filtered_df.columns:
    display_columns.append("lot_id")

display_columns += [
    "risk_score_percent",
    "warning_level",
    "root_cause_summary",
    "recommended_action"
]


# 16. Display results
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


# 17. Download results
download_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Results",
    data=download_data,
    file_name="yieldguard_filtered_results.csv",
    mime="text/csv"
)


# 18. Inspect one wafer lot
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
        df["lot_id"].astype(str)
        == selected_lot
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
            int(
                selected_row[
                    "predicted_risk_label"
                ]
            )
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
        column
        for column in process_columns
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


# 19. MVP disclaimer
st.divider()

st.caption(
    "MVP disclaimer: This dashboard currently uses synthetic "
    "semiconductor-style data and rule-based explanations. "
    "The identified factors are possible risk indicators, "
    "not confirmed physical root causes."
)