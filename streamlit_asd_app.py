import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.getcwd()

model = joblib.load(os.path.join(BASE_DIR, "asd_best_model.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "asd_shap_explainer.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "asd_feature_columns.pkl"))

raw_df = pd.read_csv(os.path.join(BASE_DIR, "autism_screening.csv"))

st.set_page_config(page_title="ASD Prediction System", layout="wide")

st.title("Autism Spectrum Disorder Prediction System")
st.write("Enter behavioural and demographic information to predict ASD status.")

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
    options = sorted(raw_df[raw_feature].dropna().astype(str).unique())

    selected = st.selectbox(
        label,
        options,
        key=model_feature
    )

    encoder = LabelEncoder()
    encoder.fit(raw_df[raw_feature].dropna().astype(str))

    input_data[model_feature] = int(encoder.transform([selected])[0])


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

    predicted_class = "ASD" if prediction == 1 else "Non-ASD"

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

    shap_df["Absolute Contribution"] = abs(shap_df["SHAP Contribution"])
    shap_df = shap_df.sort_values(
        by="Absolute Contribution",
        ascending=False
    )

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Class", predicted_class)
    c2.metric("ASD Probability", f"{round(probability * 100, 2)}%")
    c3.metric("Confidence Level", confidence)

    st.subheader("Top Behavioural Indicators")
    st.dataframe(shap_df.head(5), use_container_width=True)
