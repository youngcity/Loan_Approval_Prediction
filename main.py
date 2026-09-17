import joblib
import pandas as pd
import streamlit as st

# Function to load the saved model and perform inference


@st.cache_resource
def load_and_predict(features_df):
  # Load the trained model
  model = joblib.load("loan_approval_model.pki")

  # Generate predictions
  prediction = model.predict(features_df)
  probability = model.predict_proba(features_df)

  return prediction[0], probability[0][1]


# Streamlit UI Configuration
st.set_page_config(
    page_title="Loan Approval Predictor", page_icon="🏦", layout="centered"
)

st.title("🏦 Loan Approval Prediction App")
st.write(
    "Enter applicant details below to check loan eligibility in real time."
)

st.divider()

# Input Form
with st.form("loan_application_form"):
  st.subheader("Applicant Details")

  col1, col2 = st.columns(2)

  with col1:
    loan_id = st.text_input("Loan ID", value="LP001001")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["No", "Yes"])

  with col2:
    applicant_income = st.number_input(
        "Applicant Income ($)", min_value=0, value=5000, step=500
    )
    coapplicant_income = st.number_input(
        "Coapplicant Income ($)", min_value=0, value=2000, step=500
    )
    loan_amount = st.number_input(
        "Loan Amount (in Thousands)", min_value=1, value=150, step=10
    )
    loan_term = st.selectbox(
        "Loan Amount Term (Months)",
        options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
        index=8,
    )
    credit_history = st.selectbox(
        "Credit History",
        options=["Good (1.0)", "Bad (0.0)"],
        format_func=lambda x: x,
    )
    property_area = st.selectbox(
        "Property Area", options=["Urban", "Semiurban", "Rural"]
    )

  submit_btn = st.form_submit_button("Predict Loan Status", use_container_width=True)

# Process form input on submit
if submit_btn:
  # Map human-readable inputs to the encoded values expected by your model
  gender_encoded = 1 if gender == "Male" else 0
  married_encoded = 1 if married == "Yes" else 0
  dependents_encoded = (
      3 if dependents == "3+" else int(dependents)
  )  # Maps '0','1','2','3+' to integers
  education_encoded = 0 if education == "Graduate" else 1
  self_employed_encoded = 1 if self_employed == "Yes" else 0
  credit_history_encoded = 1.0 if "Good" in credit_history else 0.0

  property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
  property_area_encoded = property_map[property_area]

  # Assemble the DataFrame matching the exact input format of the trained model
  input_data = pd.DataFrame({
      "Gender": [gender_encoded],
      "Married": [married_encoded],
      "Dependents": [dependents_encoded],
      "Education": [education_encoded],
      "Self_Employed": [self_employed_encoded],
      "ApplicantIncome": [applicant_income],
      "CoapplicantIncome": [coapplicant_income],
      "LoanAmount": [loan_amount],
      "Loan_Amount_Term": [loan_term],
      "Credit_History": [credit_history_encoded],
      "Property_Area": [property_area_encoded],
  })

  # Call prediction function
  try:
    pred_status, pred_prob = load_and_predict(input_data)

    st.divider()

    if pred_status == 1:
      st.success(
          f"🎉 **Loan Status: Approved**\n\nProbability of Approval:"
          f" **{pred_prob * 100:.1f}%**"
      )
    else:
      st.error(
          f"❌ **Loan Status: Rejected**\n\nProbability of Approval:"
          f" **{pred_prob * 100:.1f}%**"
      )

  except Exception as e:
    st.error(f"Error making prediction: {e}")