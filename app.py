import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="CreditWise Loan System", page_icon="💳", layout="centered")

st.title("💳 CreditWise Loan Approval System")
st.write(
    "This tool predicts whether a loan application is likely to be **Approved** or "
    "**Rejected**, based on a machine learning model (Logistic Regression) trained on "
    "SecureTrust Bank's historical loan data."
)

# ----------------------------
# Load model + preprocessing artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer_values = joblib.load("imputer_values.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, imputer_values, encoders

model, scaler, imputer_values, encoders = load_artifacts()

le_edu = encoders["label_encoder_education"]
ohe = encoders["ohe"]
ohe_cols = encoders["ohe_cols"]
final_columns = encoders["final_columns"]

# ----------------------------
# Input form
# ----------------------------
st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=10000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (monthly)", min_value=0, value=0, step=500)
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    savings = st.number_input("Savings Balance", min_value=0, value=20000, step=500)
    collateral_value = st.number_input("Collateral Value", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=15000, step=500)

with col2:
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business", "Car"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    education_level = st.selectbox("Education Level", list(le_edu.classes_))
    gender = st.selectbox("Gender", ["Male", "Female"])
    employer_category = st.selectbox("Employer Category", ["Government", "Private", "MNC", "Business", "Unemployed"])

st.markdown("---")

# ----------------------------
# Predict
# ----------------------------
if st.button("Check Loan Eligibility", type="primary", use_container_width=True):

    # Build a single-row dataframe matching the raw dataset schema
    raw_input = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    # Encode Education_Level using the saved LabelEncoder
    raw_input["Education_Level"] = le_edu.transform(raw_input["Education_Level"])

    # One-hot encode the remaining categorical columns using the saved encoder
    encoded = ohe.transform(raw_input[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=raw_input.index)
    processed = pd.concat([raw_input.drop(columns=ohe_cols), encoded_df], axis=1)

    # Align columns exactly with what the model was trained on
    processed = processed.reindex(columns=final_columns, fill_value=0)

    # Scale
    scaled_input = scaler.transform(processed)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    st.markdown("### Result")
    if prediction == 1:
        st.success(f"✅ Loan Likely **Approved** (confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"❌ Loan Likely **Rejected** (confidence: {probability[0]*100:.1f}%)")

    st.caption(
        "This is an automated prediction based on historical data and is meant to assist, "
        "not replace, final human verification."
    )

st.markdown("---")
st.caption("CreditWise Loan System — ML Engineering Project for SecureTrust Bank")
