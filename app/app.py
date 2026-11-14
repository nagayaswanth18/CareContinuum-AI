
import streamlit as st
import pandas as pd
import json
import os
from catboost import CatBoostClassifier

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CareContinuum AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(120deg, #123B50, #187B80);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E2F2F2;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 5px;
        color: #187B80;
    }

    .result-card {
        padding: 24px;
        border-radius: 14px;
        background: #EAF5F3;
        border-left: 6px solid #187B80;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-card h2 {
        margin: 0;
        font-size: 27px;
        color: #145B55;
    }

    .result-card p {
        margin-top: 8px;
        color: #365B58;
    }

    .small-note {
        color: #777777;
        font-size: 13px;
    }

    div[data-testid="stForm"] {
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #DDE6E8;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #187B80;
        color: white;
        border-radius: 9px;
        border: none;
        min-height: 45px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #125D63;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD MODEL AND METADATA
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(
        os.path.join(
            MODEL_DIR,
            "carecontinuum_catboost_10features.cbm"
        )
    )
    return model


model = load_model()

with open(
    os.path.join(MODEL_DIR, "model_metadata_10features.json")
) as f:
    metadata = json.load(f)

with open(
    os.path.join(MODEL_DIR, "category_options_10features.json")
) as f:
    category_options = json.load(f)

with open(
    os.path.join(MODEL_DIR, "numeric_stats.json")
) as f:
    numeric_stats = json.load(f)

feature_columns = metadata["feature_columns"]
categorical_features = metadata["categorical_features"]


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🏥 CareContinuum AI")

    st.caption("Hospital Readmission Prediction")

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** CatBoost Classifier")
    st.write("**Model:** Compact 10-feature model")
    st.write("**Number of inputs:** 10")
    st.write("**Prediction threshold:** 0.50")

    st.divider()

    st.subheader("Test Performance")

    st.metric("Test Accuracy", "63.46%")
    st.metric("Readmitted F1 Score", "0.5836")
    st.metric("Readmitted Recall", "54.49%")

    st.divider()

    st.caption(
        "Developed as an educational machine learning project."
    )


# --------------------------------------------------
# 5. MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🏥 CareContinuum AI</h1>
    <p>
        Intelligent Hospital Readmission Estimation
    </p>
    <p>
        A machine learning prototype using hospital
        encounter information and CatBoost classification.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 6. PROJECT OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Project Overview</div>',
    unsafe_allow_html=True
)

st.write(
    "CareContinuum AI uses a trained machine learning model "
    "to estimate the probability of readmission based on "
    "10 selected patient and hospital encounter features."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Input Features", "10")

with col2:
    st.metric("Test Accuracy", "63.46%")

with col3:
    st.metric("Model Algorithm", "CatBoost")

st.divider()


# --------------------------------------------------
# 7. PATIENT INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Patient Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the following encounter details to generate "
    "a model-based readmission estimate."
)

labels = {
    "number_inpatient": "Previous Inpatient Visits",
    "diag_1": "Primary Diagnosis",
    "admission_source_id": "Admission Source",
    "diag_2": "Secondary Diagnosis",
    "number_diagnoses": "Number of Diagnoses",
    "medical_specialty": "Medical Specialty",
    "admission_type_id": "Admission Type",
    "num_medications": "Number of Medications",
    "number_emergency": "Previous Emergency Visits",
    "age": "Age Group"
}

with st.form("patient_form"):

    input_values = {}

    left_col, right_col = st.columns(2)

    for i, feature in enumerate(feature_columns):

        current_col = (
            left_col if i % 2 == 0 else right_col
        )

        with current_col:

            label = labels.get(feature, feature)

            if feature in categorical_features:

                options = category_options[feature]

                default_index = (
                    options.index("Unknown")
                    if "Unknown" in options
                    else 0
                )

                input_values[feature] = st.selectbox(
                    label,
                    options=options,
                    index=default_index
                )

            else:

                stats = numeric_stats[feature]

                input_values[feature] = st.number_input(
                    label,
                    min_value=int(stats["min"]),
                    max_value=int(stats["max"]),
                    value=int(stats["median"]),
                    step=1
                )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Generate Readmission Estimate",
        use_container_width=True
    )


# --------------------------------------------------
# 8. PREDICTION RESULT
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    probability = model.predict_proba(input_df)[0][1]

    prediction = int(
        probability >= metadata["prediction_threshold"]
    )

    st.divider()

    st.markdown(
        '<div class="section-heading">Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.write("Model-generated probability estimates:")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Readmission Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Not Readmission Probability",
            f"{(1 - probability) * 100:.2f}%"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.markdown("""
        <div class="result-card">
            <h2>⚠️ Model Prediction: Readmitted</h2>
            <p>
                The model's estimated probability is at or above
                its configured classification threshold.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <h2>✓ Model Prediction: Not Readmitted</h2>
            <p>
                The model's estimated probability is below
                its configured classification threshold.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("View submitted patient information"):
        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "The classification is based on a 0.50 probability "
        "threshold. It is an educational model estimate, "
        "not a clinically validated prediction or medical advice."
    )


# --------------------------------------------------
# 9. ABOUT THE PROJECT
# --------------------------------------------------

st.divider()

with st.expander("About CareContinuum AI"):

    st.write(
        "CareContinuum AI is an educational machine learning "
        "project developed using the Diabetes 130-US Hospitals "
        "dataset (1999–2008)."
    )

    st.write(
        "The final compact model uses 10 selected features "
        "and a CatBoost binary classification algorithm."
    )

    st.write(
        "The reported test accuracy is 63.46%, with a "
        "readmitted-class F1 score of 0.5836."
    )

    st.write(
        "The model is not clinically validated and must "
        "not be used for actual medical decisions."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p style="text-align:center; color:grey;">
        <b>CareContinuum AI</b><br>
        Educational Machine Learning Project<br>
        Compact 10-Feature CatBoost Classification
    </p>
    """,
    unsafe_allow_html=True
)
