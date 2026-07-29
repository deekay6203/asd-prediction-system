import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "asd_best_model.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "asd_shap_explainer.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "asd_feature_columns.pkl"))
label_encoders = joblib.load(
    os.path.join(BASE_DIR, "asd_label_encoders.pkl")
)
raw_df = pd.read_csv(os.path.join(BASE_DIR, "autism_screening.csv"))

st.set_page_config(page_title="ASD Prediction System", layout="wide")

st.title("Autism Spectrum Disorder Prediction System")
st.write("Enter behavioural and demographic information to predict ASD status.")
st.markdown("""
<style>

/* Use almost the full browser width */
.block-container {
    max-width: none !important;
    width: 96% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
}

/* Larger title */
h1 {
    font-size: 34px !important;
    line-height: 1.2 !important;
}

/* Section headings */
h3 {
    font-size: 22px !important;
    margin-top: 18px;
}

/* Labels */
label {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Input height */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    min-height: 40px;
}

/* Predict button */
.stButton > button {
    height: 44px;
    font-size: 16px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

DISPLAY_NAMES = {
    "A1_Score": "A1 Score",
    "A2_Score": "A2 Score",
    "A3_Score": "A3 Score",
    "A4_Score": "A4 Score",
    "A5_Score": "A5 Score",
    "A6_Score": "A6 Score",
    "A7_Score": "A7 Score",
    "A8_Score": "A8 Score",
    "A9_Score": "A9 Score",
    "A10_Score": "A10 Score",
    "age": "Age",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "jundice": "Jaundice at Birth",
    "Jaundice": "Jaundice at Birth",
    "austim": "Family History of Autism",
    "Autism": "Family History of Autism",
    "Austim": "Family History of Autism",
    "contry_of_res": "Country of Residence",
    "Country_of_Residence": "Country of Residence",
    "used_app_before": "Previously Used ASD Screening App",
    "relation": "Relationship to Individual"
}

RAW_COLUMN_MAP = {
    "Jaundice": "jundice",
    "jundice": "jundice",
    "Autism": "austim",
    "Austim": "austim",
    "austim": "austim",
    "Country_of_Residence": "contry_of_res",
    "contry_of_res": "contry_of_res",
    "gender": "gender",
    "ethnicity": "ethnicity",
    "used_app_before": "used_app_before",
    "relation": "relation"
}

input_data = {}

def display_name(feature):
    return DISPLAY_NAMES.get(feature, feature.replace("_", " "))

def encode_dropdown(model_feature, raw_feature, label):
    encoder = label_encoders[model_feature]

    options = [
        str(value)
        for value in encoder.classes_
        if str(value) != "?"
    ]

    display_map = {
        "f": "Female",
        "m": "Male",
        "yes": "Yes",
        "no": "No"
    }

    display_options = [
        display_map.get(option.lower(), option)
        for option in options
    ]

    selected_display = st.selectbox(
        label,
        display_options,
        key=model_feature
    )

    reverse_map = {
        display_map.get(option.lower(), option): option
        for option in options
    }

    selected_value = reverse_map[selected_display]

    input_data[model_feature] = int(
        encoder.transform([selected_value])[0]
    )

st.subheader("Behavioural Screening Questions")

score_cols = [
    col for col in feature_columns
    if col.startswith("A") and col.endswith("_Score")
]

score_cols_layout = st.columns(5)

for i, feature in enumerate(score_cols):
    with score_cols_layout[i % 5]:
        answer = st.selectbox(
            display_name(feature),
            ["No", "Yes"],
            key=feature
        )
        input_data[feature] = 1 if answer == "Yes" else 0


st.subheader("Demographic Information")

demo_cols = st.columns(3)

for feature in feature_columns:
    if feature in input_data:
        continue

    label = display_name(feature)

    if feature == "age":
        with demo_cols[0]:
            input_data[feature] = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=5,
                step=1
            )

    elif feature in RAW_COLUMN_MAP:
        raw_feature = RAW_COLUMN_MAP[feature]
        with demo_cols[len(input_data) % 3]:
            encode_dropdown(feature, raw_feature, label)

    else:
        with demo_cols[len(input_data) % 3]:
            input_data[feature] = st.number_input(
                label,
                value=0.0,
                step=1.0
            )


input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]

st.markdown("---")

left, center, right = st.columns([3, 2, 3])

with center:
    predict = st.button(
        "Predict ASD",
        use_container_width=True,
    )

if predict:

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    predicted_class = (
    "ASD Detected"
    if prediction == 1
    else "ASD Not Detected"
    )

    predicted_confidence = (
        probability if prediction == 1
        else 1 - probability
    )

    if predicted_confidence >= 0.80:
        confidence = "High"
    elif predicted_confidence >= 0.60:
        confidence = "Moderate"
    else:
        confidence = "Low"

    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "Feature": [display_name(col) for col in input_df.columns],
        "Feature Value": input_df.iloc[0].values,
        "SHAP Contribution": shap_values.values[0]
    })

    shap_df["Absolute Contribution"] = abs(
        shap_df["SHAP Contribution"]
    )

    shap_df["SHAP Contribution"] = (
        shap_df["SHAP Contribution"].round(3)
    )

    shap_df["Absolute Contribution"] = (
        shap_df["Absolute Contribution"].round(3)
    )

    shap_df = shap_df.sort_values(
        by="Absolute Contribution",
        ascending=False
    )

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Class",
        predicted_class
    )

    c2.metric(
        "Predicted ASD Risk",
        f"{probability * 100:.2f}%"
    )

    c3.metric(
        "Confidence Level",
        confidence
    )

    st.subheader("Top Behavioural Indicators")

    st.dataframe(
        shap_df.head(5),
        use_container_width=True
    )

    st.info(
        "This application is intended for research and educational purposes only "
        "and should not be used as a substitute for professional clinical diagnosis."
    )
