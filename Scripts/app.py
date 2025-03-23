import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings("ignore")

# ------------------------ Load Saved Model & Preprocessing Objects ------------------------

#print(sklearn.__version__) 

# Load trained model
model = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/best_model.pkl", "rb"))

# Load scaler & encoders
scaler = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/scaler.pkl", "rb"))
encoders = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/label_encoders.pkl", "rb"))
income_lambda,premium_lambda = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/boxcox_lambdas.pkl", "rb"))  # (income_lambda, premium_lambda)

scale_factor = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/scale_factor.pkl", "rb"))


#print("Premium Lambda:", premium_lambda)

# ------------------------ Define Function for Preprocessing New Input ------------------------
def bin_features(df):
    """Apply binning to numerical features"""
    feature_bins = {
        'Age': [18, 30, 40, 50, 64, float('inf')],
        'Number of Dependents': [0, 1, 2, 3, float('inf')],
        'Health Score': [0, 15, 25, 35, float('inf')],
        'Previous Claims': [0, 1, 2, float('inf')],
        'Vehicle Age': [0, 5, 10, 20, float('inf')],
        'Credit Score': [0, 300, 600, 800, float('inf')],
        'Insurance Duration': [0, 3, 6, 9, float('inf')],
    }
    
    for feature, bins in feature_bins.items():
        if feature in df.columns:
            df[feature] = pd.cut(df[feature], bins=bins, labels=False, right=True, include_lowest=True)
            df[feature].fillna(-1, inplace=True)  # ✅ Fix: Fill missing values before converting to int
            df[feature] = df[feature].astype(int)
    return df

def preprocess_input(data,type="form"):
    """ Preprocess new input data before prediction """

    df=pd.DataFrame()

    if type=="form":
        df = pd.DataFrame([data])
    else:
        df = data.copy()
        df.drop(columns=['Policy Start Date', 'id'], errors="ignore", inplace=True)  # ✅ 'id' is saved separately

    # Fill missing values
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    for feature in numerical_features:
        df[feature].fillna(df[feature].mean(), inplace=True)

    for feature in categorical_features:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

    # Apply binning
    df = bin_features(df)

    # Mapping ordered categorical features
    mapping_definitions = {
        "Education Level": {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3},
        "Customer Feedback": {"Poor": 0, "Average": 1, "Good": 2},
        "Exercise Frequency": {"Rarely": 0, "Weekly": 1, "Monthly": 2, "Daily": 3},
        "Policy Type": {"Basic": 0, "Comprehensive": 1, "Premium": 2}
    }

    for column, mapping in mapping_definitions.items():
        df[column] = df[column].map(mapping).fillna(-1)  # Handle unseen categories

    # Encode categorical features using stored LabelEncoders
    for col, le in encoders.items():
        if col in df:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = -1  # Assign unknown categories a default value (-1)

    # Apply Box-Cox transformation for 'Annual Income'
    df['Annual Income'] = stats.boxcox(df['Annual Income'] + 1, lmbda=income_lambda)

    # Scale numerical values
    df[['Annual Income']] = scaler.transform(df[['Annual Income']])

    return df

# ------------------------ Streamlit UI ------------------------

st.title("💰 Insurance Premium Predictor")


st.write("Upload a CSV file with customer details to predict their insurance premium.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read uploaded CSV
    test_df = pd.read_csv(uploaded_file)

    # Store original ID column
    test_ids = test_df["id"]

    # Preprocess the data
    X_test = preprocess_input(test_df,"uploads")

    # Predict insurance premiums
    predicted_premiums = model.predict(X_test)

    # Apply inverse Box-Cox transformation
    predicted_premiums = inv_boxcox(predicted_premiums, premium_lambda)

    # Create result dataframe
    results_df = pd.DataFrame({"id": test_ids, "Predicted Premium": predicted_premiums})

    # Save results as CSV
    results_csv = results_df.to_csv(index=False).encode("utf-8")

    # Provide download button
    st.download_button(
        label="📥 Download Predictions",
        data=results_csv,
        file_name="predicted_premiums.csv",
        mime="text/csv",
    )

    st.success("✅ Predictions completed! Download the file above.")

st.write("Enter the customer details to estimate their insurance premium.")
# Create form for user input
with st.form("insurance_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=41)
    gender = st.selectbox("Gender", ["Male", "Female"],placeholder="Male")
    income = st.number_input("Annual Income (₹)", value=32745.21)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"],placeholder="Single")
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=4, value=2)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"],placeholder="Master's")
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"],placeholder="Employed")
    health_score = st.number_input("Health Score", value=25.7)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"],placeholder="Suburban")
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"],placeholder="Premium")
    prev_claims = st.number_input("Previous Claims", min_value=0, max_value=9, value=1)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=19, value=9)
    credit_score = st.number_input("Credit Score", max_value=900, value=592)
    insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=5)
    customer_feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"],placeholder="Average")
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"],placeholder="Yes")
    exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"],placeholder="Weekly")
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"],placeholder="House")

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
    processed_input = preprocess_input(user_data,"form")

    # Make prediction
    predicted_premium = model.predict(processed_input)

    predicted_premium = inv_boxcox(predicted_premium, premium_lambda) * scale_factor

    # Display the predicted premium amount
    st.success(f"💰 Estimated Insurance Premium: ₹{predicted_premium[0]:,.2f}")
