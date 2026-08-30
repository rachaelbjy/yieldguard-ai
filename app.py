import os

import joblib
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)


# Page setup
st.set_page_config(
    page_title="YieldGuard AI",
    page_icon="🧠",
    layout="wide",
)

st.title("YieldGuard AI")
st.caption(
    "Predict wafer-lot yield risk, explain AI decision drivers, "
    "and surface process conditions that may require attention."
)

st.info(
    "Upload a wafer-lot CSV from the sidebar to run live predictions, "
    "or explore the built-in demo dataset."
)


# Load model
model_path = "models/yieldguard_model.pkl"
demo_data_path = "data/wafer_data.csv"

if not os.path.exists(model_path):
    st.error(
        "Trained model not found. Please run train_model.py first."
    )
    st.stop()

model_package = joblib.load(model_path)

model = model_package["model"]
features = model_package["features"]

class_list = list(model.classes_)

if set(class_list) != {0, 1}:
    st.error(
        "The trained model must use class 0 for low risk "
        "and class 1 for high risk."
    )
    st.stop()

high_risk_position = class_list.index(1)

shap_explainer = shap.TreeExplainer(model)

feature_display_names = {
    "chamber_temp": "Chamber Temperature",
    "chamber_pressure": "Chamber Pressure",
    "etch_time": "Etch Time",
    "film_thickness": "Film Thickness",
    "defect_count": "Defect Count",
}

missing_process_features = [
    feature for feature in feature_display_names
    if feature not in features
]

if missing_process_features:
    st.error(
        "The trained model is missing process features required "
        f"by the dashboard: {missing_process_features}"
    )
    st.stop()


# Helper functions
def get_warning_level(score):
    if score >= 0.70:
        return "HIGH RISK"

    if score >= 0.40:
        return "MEDIUM RISK"

    return "LOW RISK"


def analyse_risk_factors(row):
    factors = []
    actions = []

    if row["defect_count"] >= 15:
        factors.append("High defect count")
        actions.append(
            "Review inspection results and defect source"
        )

    if row["chamber_temp"] > 90:
        factors.append("Chamber temperature too high")
        actions.append(
            "Check chamber temperature control"
        )

    if row["chamber_pressure"] > 2.6:
        factors.append("Chamber pressure too high")
        actions.append(
            "Inspect pressure control and vacuum system"
        )

    if row["etch_time"] > 65:
        factors.append("Etch time too long")
        actions.append(
            "Review etch recipe and process duration"
        )

    if row["film_thickness"] > 110:
        factors.append("Film thickness too high")
        actions.append(
            "Check deposition rate and recipe settings"
        )

    elif row["film_thickness"] < 90:
        factors.append("Film thickness too low")
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


def run_predictions(input_df):
    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required columns: {missing_features}"
        )

    results = input_df.copy()

    for feature in features:
        results[feature] = pd.to_numeric(
            results[feature],
            errors="coerce",
        )

    if results[features].isnull().any().any():
        raise ValueError(
            "Required feature columns contain missing "
            "or non-numeric values."
        )

    if (
        results[features]
        .isin([float("inf"), float("-inf")])
        .any()
        .any()
    ):
        raise ValueError(
            "Required feature columns contain infinite values."
        )

    X_batch = results[features]

    predictions = model.predict(X_batch)

    all_probabilities = model.predict_proba(
        X_batch
    )

    risk_scores = all_probabilities[
        :,
        high_risk_position,
    ]

    results["predicted_risk_label"] = predictions
    results["risk_score"] = risk_scores

    results["risk_score_percent"] = (
        risk_scores * 100
    ).round(2)

    results["warning_level"] = (
        results["risk_score"].apply(
            get_warning_level
        )
    )

    explanations = results.apply(
        analyse_risk_factors,
        axis=1,
    )

    return pd.concat(
        [results, explanations],
        axis=1,
    )


# Data input
st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload wafer-lot CSV",
    type=["csv"],
)

using_uploaded_data = uploaded_file is not None

if using_uploaded_data:
    data_source = uploaded_file

else:
    if not os.path.exists(demo_data_path):
        st.error(
            "Built-in demo data not found. "
            "Please run generate_data.py first or upload a wafer-lot CSV."
        )
        st.stop()

    data_source = demo_data_path


try:
    input_df = pd.read_csv(
        data_source
    )

except (
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    UnicodeDecodeError,
):
    st.error(
        "The selected file could not be read as a valid CSV."
    )
    st.stop()


if input_df.empty:
    st.error(
        "The CSV contains no wafer lots."
    )
    st.stop()


try:
    df = run_predictions(
        input_df
    )

except ValueError as error:
    st.error(
        str(error)
    )
    st.stop()


if using_uploaded_data:
    lot_word = (
        "wafer lot"
        if len(df) == 1
        else "wafer lots"
    )

    st.success(
        f"Successfully analysed {len(df)} {lot_word}."
    )

    st.sidebar.success(
        "Live Prediction Mode: analysing uploaded wafer-lot data."
    )

    st.sidebar.caption(
        f"Current file: {uploaded_file.name}"
    )

else:
    st.sidebar.info(
        "Demo Mode: analysing the built-in synthetic wafer dataset."
    )


# Model performance for uploaded labelled data
show_model_performance = (
    using_uploaded_data
    and "risk_label" in df.columns
)

if show_model_performance:
    actual_labels = pd.to_numeric(
        df["risk_label"],
        errors="coerce",
    )

    valid_labels = (
        not actual_labels.isnull().any()
        and set(
            actual_labels.unique()
        ).issubset({0, 1})
    )

    if valid_labels:
        actual_labels = (
            actual_labels.astype(int)
        )

        model_accuracy = accuracy_score(
            actual_labels,
            df["predicted_risk_label"],
        )

        model_precision = precision_score(
            actual_labels,
            df["predicted_risk_label"],
            zero_division=0,
        )

        model_recall = recall_score(
            actual_labels,
            df["predicted_risk_label"],
            zero_division=0,
        )

        cm = confusion_matrix(
            actual_labels,
            df["predicted_risk_label"],
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

    else:
        show_model_performance = False

        st.warning(
            "Model performance is not shown because "
            "risk_label must contain only 0 and 1."
        )


# Production overview
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

st.subheader(
    "Production Risk Overview"
)

metric_1, metric_2, metric_3, metric_4, metric_5 = (
    st.columns(5)
)

metric_1.metric(
    "Total Wafer Lots",
    total_lots,
)

metric_2.metric(
    "High Risk",
    high_risk_count,
)

metric_3.metric(
    "Medium Risk",
    medium_risk_count,
)

metric_4.metric(
    "Low Risk",
    low_risk_count,
)

metric_5.metric(
    "Average Risk Score",
    f"{average_risk_score:.1f}%",
)

st.caption(
    "Warning thresholds: HIGH RISK ≥ 70% | "
    "MEDIUM RISK 40% to < 70% | "
    "LOW RISK < 40%. "
    "Binary model labels and dashboard warning levels "
    "use different decision thresholds."
)

st.divider()


# Model evaluation
if show_model_performance:
    st.subheader(
        "Model Performance"
    )

    performance_1, performance_2, performance_3 = (
        st.columns(3)
    )

    performance_1.metric(
        "Accuracy",
        f"{model_accuracy:.1%}",
    )

    performance_2.metric(
        "Precision",
        f"{model_precision:.1%}",
    )

    performance_3.metric(
        "Recall",
        f"{model_recall:.1%}",
    )

    st.markdown(
        "#### Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=[
            "Actual Low Risk",
            "Actual High Risk",
        ],
        columns=[
            "Predicted Low Risk",
            "Predicted High Risk",
        ],
    )

    st.dataframe(
        confusion_df,
        width="stretch",
    )

    st.caption(
        "For meaningful model evaluation, use held-out labelled data "
        "that was not used during model training."
    )

    st.divider()


# Warning distribution
st.subheader(
    "Warning Level Distribution"
)

risk_order = [
    "HIGH RISK",
    "MEDIUM RISK",
    "LOW RISK",
]

risk_distribution_df = pd.DataFrame({
    "Warning Level": risk_order,
    "Count": [
        int(
            (
                df["warning_level"] == level
            ).sum()
        )
        for level in risk_order
    ],
})

st.vega_lite_chart(
    risk_distribution_df,
    {
        "mark": {
            "type": "bar"
        },
        "encoding": {
            "x": {
                "field": "Warning Level",
                "type": "nominal",
                "sort": risk_order,
                "axis": {
                    "labelAngle": 0
                },
            },
            "y": {
                "field": "Count",
                "type": "quantitative",
                "title": "Number of Wafer Lots",
            },
        },
    },
    width="stretch",
)

st.divider()


# Filters and result table
st.sidebar.header(
    "Filters"
)

selected_warning_levels = (
    st.sidebar.multiselect(
        "Warning Level",
        options=risk_order,
        default=risk_order,
    )
)

minimum_risk_score = (
    st.sidebar.slider(
        "Minimum Risk Score",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )
)

filtered_df = df[
    df["warning_level"].isin(
        selected_warning_levels
    )
    & (
        df["risk_score_percent"]
        >= minimum_risk_score
    )
].copy()

filtered_df = (
    filtered_df.sort_values(
        "risk_score_percent",
        ascending=False,
    )
)

display_columns = []

if "lot_id" in filtered_df.columns:
    display_columns.append(
        "lot_id"
    )

display_columns += [
    "risk_score_percent",
    "warning_level",
    "root_cause_summary",
    "recommended_action",
]

display_df = (
    filtered_df[
        display_columns
    ]
    .rename(
        columns={
            "lot_id": "Lot ID",
            "risk_score_percent": "Risk Score (%)",
            "warning_level": "Warning Level",
            "root_cause_summary": "Possible Risk Factors",
            "recommended_action": "Recommended Action",
        }
    )
)

st.subheader(
    "Wafer Lot Risk Results"
)

st.write(
    f"Showing {len(filtered_df)} of "
    f"{len(df)} wafer lots"
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)

download_data = (
    filtered_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)

st.download_button(
    label="Download Filtered Results",
    data=download_data,
    file_name="yieldguard_filtered_results.csv",
    mime="text/csv",
)

st.divider()


# Inspect one wafer lot
st.subheader(
    "Inspect One Wafer Lot"
)

if "lot_id" in df.columns:
    inspection_df = (
        df.sort_values(
            "risk_score_percent",
            ascending=False,
        )
        .copy()
    )

    inspection_df["lot_id"] = (
        inspection_df["lot_id"]
        .astype(str)
    )

    if (
        inspection_df["lot_id"]
        .duplicated()
        .any()
    ):
        st.warning(
            "Duplicate lot_id values were found. "
            "Individual inspection uses the first matching row."
        )

    lot_options = (
        inspection_df["lot_id"]
        .tolist()
    )

    selected_lot = st.selectbox(
        "Select Wafer Lot",
        options=lot_options,
    )

    selected_row = inspection_df[
        inspection_df["lot_id"]
        == selected_lot
    ].iloc[0]

    detail_1, detail_2, detail_3 = (
        st.columns(3)
    )

    detail_1.metric(
        "Risk Score",
        f"{selected_row['risk_score_percent']:.1f}%",
    )

    detail_2.metric(
        "Warning Level",
        selected_row["warning_level"],
    )

    predicted_label = int(
        selected_row[
            "predicted_risk_label"
        ]
    )

    predicted_class_text = (
        "HIGH RISK (1)"
        if predicted_label == 1
        else "LOW RISK (0)"
    )

    detail_3.metric(
        "Predicted Class",
        predicted_class_text,
    )

    st.markdown(
        "#### Process Conditions"
    )

    process_columns = [
        feature for feature in features
        if feature in selected_row.index
    ]

    process_details = pd.DataFrame({
        "Feature": [
            feature_display_names.get(
                feature,
                feature,
            )
            for feature in process_columns
        ],
        "Value": [
            selected_row[feature]
            for feature in process_columns
        ],
    })

    st.dataframe(
        process_details,
        width="stretch",
        hide_index=True,
    )

    selected_input = pd.DataFrame(
        [
            selected_row[
                features
            ].to_dict()
        ]
    )[features]

    shap_values = shap_explainer(
        selected_input
    )

    if shap_values.values.ndim == 3:
        high_risk_shap_values = (
            shap_values.values[
                0,
                :,
                high_risk_position,
            ]
        )

    elif shap_values.values.ndim == 2:
        high_risk_shap_values = (
            shap_values.values[0]
        )

    else:
        st.error(
            "Unexpected SHAP output shape."
        )
        st.stop()

    if (
        len(high_risk_shap_values)
        != len(features)
    ):
        st.error(
            "SHAP output does not match "
            "the model feature list."
        )
        st.stop()

    shap_df = pd.DataFrame({
        "Feature": features,
        "Value": selected_input.iloc[0].values,
        "SHAP Contribution": high_risk_shap_values,
    })

    # Use the original SHAP values for ranking and direction
    shap_df["Absolute Impact"] = (
        shap_df[
            "SHAP Contribution"
        ].abs()
    )

    shap_df["Direction"] = (
        shap_df[
            "SHAP Contribution"
        ].apply(
            lambda value: (
                "Pushes toward HIGH RISK"
                if value > 0
                else (
                    "Pushes toward LOW RISK"
                    if value < 0
                    else "Neutral"
                )
            )
        )
    )

    shap_df = (
        shap_df.sort_values(
            "Absolute Impact",
            ascending=False,
        )
    )

    shap_df["Feature"] = (
        shap_df["Feature"]
        .replace(
            feature_display_names
        )
    )

    top_shap_feature = (
        shap_df.iloc[0]
    )

    shap_display_df = shap_df[
        [
            "Feature",
            "Value",
            "SHAP Contribution",
            "Direction",
        ]
    ].copy()

    shap_display_df[
        "SHAP Contribution"
    ] = (
        shap_display_df[
            "SHAP Contribution"
        ].round(4)
    )

    st.markdown(
        "#### AI Model Explanation (SHAP)"
    )

    st.dataframe(
        shap_display_df,
        width="stretch",
        hide_index=True,
    )

    shap_chart_df = (
        shap_display_df[
            [
                "Feature",
                "SHAP Contribution",
            ]
        ]
        .copy()
    )

    shap_feature_order = (
        shap_chart_df[
            "Feature"
        ].tolist()
    )

    st.vega_lite_chart(
        shap_chart_df,
        {
            "mark": {
                "type": "bar"
            },
            "encoding": {
                "y": {
                    "field": "Feature",
                    "type": "nominal",
                    "sort": shap_feature_order,
                    "axis": {
                        "title": None,
                        "labelLimit": 200,
                    },
                },
                "x": {
                    "field": "SHAP Contribution",
                    "type": "quantitative",
                    "title": "SHAP Contribution",
                },
                "tooltip": [
                    {
                        "field": "Feature",
                        "type": "nominal",
                    },
                    {
                        "field": "SHAP Contribution",
                        "type": "quantitative",
                    },
                ],
            },
        },
        width="stretch",
    )

    st.markdown(
        "#### Key AI Driver"
    )

    st.info(
        f"{top_shap_feature['Feature']} is the strongest AI driver "
        f"for this prediction. "
        f"Current value: {top_shap_feature['Value']}. "
        f"{top_shap_feature['Direction']}."
    )

    st.markdown(
        "#### Possible Risk Factors"
    )

    st.warning(
        selected_row[
            "root_cause_summary"
        ]
    )

    st.markdown(
        "#### Recommended Action"
    )

    st.info(
        selected_row[
            "recommended_action"
        ]
    )

else:
    st.info(
        "No lot_id column is available for "
        "individual wafer-lot inspection."
    )


# MVP disclaimer
st.divider()

st.caption(
    "MVP disclaimer: This dashboard uses synthetic semiconductor-style "
    "data. SHAP explains model prediction behaviour, while rule-based "
    "process indicators and recommended actions are illustrative and "
    "do not represent confirmed physical root causes."
)