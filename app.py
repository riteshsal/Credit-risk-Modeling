import streamlit as st
import pandas as pd
import joblib

model=joblib.load('C:\\Machine Learning\\Projects\\credit risk modeling\\xgb_credit_model.pkl')
encoders={col: joblib.load(f'{col}_encoder.pkl') for col in ["Sex","Housing","Saving accounts","Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter the details of the applicant to predict the credit risk.")

age=st.number_input("Age", min_value=18, max_value=100, value=30)
sex=st.selectbox("Sex", ["male", "female"])
job=st.number_input("Job", min_value=0, max_value=3, value=1)
housing=st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts=st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_account=st.selectbox("Checking account", ["little", "moderate", "rich"])
credit_amount=st.number_input("Credit amount", min_value=0,value=1000)
duration=st.number_input("Duration", min_value=1, value=12)

input_df=pd.DataFrame({
    "Age":[age],
    "Sex":[encoders["Sex"].transform([sex])[0]],
    "Job":[job],
    "Housing":[encoders["Housing"].transform([housing])[0]],
    "Saving accounts":[encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account":[encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount":[credit_amount],
    "Duration":[duration]
})

if st.button("Predict"):
    prediction=model.predict(input_df)[0]
    if prediction==1:
        st.success("The predicted credit risk is : Good")
    else:
        st.error("The predicted credit risk is : Bad")