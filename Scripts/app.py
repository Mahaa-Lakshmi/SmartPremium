import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------------ Load Saved Model & Preprocessing Objects ------------------------

# Load trained model
model = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/backup codes/model_collab/best_model.pkl", "rb"))

# Load scaler & encoders
scaler = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/backup codes/pickles/scaler.pkl", "rb"))
encoders = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/backup codes/pickles/label_encoders.pkl", "rb"))
boxcox_lambdas = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/backup codes/pickles/boxcox_lambdas.pkl", "rb"))  # (income_lambda, premium_lambda)

income_lambda = boxcox_lambdas[0]  # Lambda used for Box-Cox transformation""

# ------------------------ Define Function for Preprocessing New Input ------------------------

def preprocess_input(data):
    """ Preprocess new input data before prediction """
    df = pd.DataFrame([data])

    # Apply Label Encoding for categorical features
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Apply Box-Cox transformation for 'Annual Income'
    df['Annual Income'] = stats.boxcox(df['Annual Income'] + 1, lmbda=income_lambda)

    # Scale numerical values
    df[['Annual Income']] = scaler.transform(df[['Annual Income']])

    return df

# ------------------------ Streamlit UI ------------------------

st.title("💰 Insurance Premium Predictor")
st.write("Enter the customer details to estimate their insurance premium.")

# Create form for user input
with st.form("insurance_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income (₹)", min_value=10000, max_value=5000000, value=50000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    health_score = st.slider("Health Score", 0, 100, 50)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    prev_claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=1)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=5)
    customer_feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    submitted = st.form_submit_button("Predict Premium")

# ------------------------ Make Prediction ------------------------

if submitted:
    # Prepare user input data
    user_data = {
        "Age": age, "Gender": gender, "Annual Income": income, "Marital Status": marital_status,
        "Number of Dependents": dependents, "Education Level": education, "Occupation": occupation,
        "Health Score": health_score, "Location": location, "Policy Type": policy_type,
        "Previous Claims": prev_claims, "Vehicle Age": vehicle_age, "Credit Score": credit_score,
        "Insurance Duration": insurance_duration, "Customer Feedback": customer_feedback,
        "Smoking Status": smoking_status, "Exercise Frequency": exercise_freq, "Property Type": property_type
    }

    # Preprocess user input
    processed_input = preprocess_input(user_data)

    # Make prediction
    predicted_premium = model.predict(processed_input)

    # Display the predicted premium amount
    st.success(f"💰 Estimated Insurance Premium: ₹{predicted_premium[0]:,.2f}")
