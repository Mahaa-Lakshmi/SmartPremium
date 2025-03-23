import pandas as pd
import numpy as np
import pickle
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load Data & Preprocessing Objects
data = pd.read_csv("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/data/train.csv")
scaler = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/scaler.pkl", "rb"))
encoders = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/label_encoders.pkl", "rb"))
income_lambda,premium_lambda = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/boxcox_lambdas.pkl", "rb"))  # (income_lambda, premium_lambda)

# Load Best Model
best_model = pickle.load(open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/best_model1.pkl", "rb"))

def preprocess_data(df, train=True):
    """ Preprocess the dataset: handle missing values, encode, transform, and scale. """
    
    # Drop irrelevant columns
    df.drop(columns=['id', 'Policy Start Date'], errors='ignore', inplace=True)
    
    # Fill missing values
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    for feature in numerical_features:
        df[feature].fillna(df[feature].mean(), inplace=True)

    for feature in categorical_features:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

    print("Binning")

    #Binning
    feature_bins = {
    'Age': [18.0, 30.0, 40.0, 50.0, 64.0, float('inf')],
    'Number of Dependents': [0.0, 1.0, 2.0, 3.0, float('inf')],
    'Health Score': [0.0, 15.0, 25.0, 35.0, float('inf')],
    'Previous Claims': [0.0, 1.0, 2.0, float('inf')],
    'Vehicle Age': [0.0, 5.0, 10.0, 20.0, float('inf')],
    'Credit Score': [0.0, 300.0, 600.0, 800.0, float('inf')],
    'Insurance Duration': [0.0, 3.0, 6.0, 9.0, float('inf')],
    }
    for feature, bins in feature_bins.items():
        df[feature] = pd.cut(df[feature], bins=bins, labels=False, right=True,include_lowest=True).astype(int)
    
    # Encode categorical variables

    #Mapping ordered features
    print("Mapping")

    mapping_definitions = {
        "Education Level": ["High School", "Bachelor's", "Master's", "PhD"],
        "Customer Feedback": ["Poor", "Average", "Good"],
        "Exercise Frequency": ["Rarely", "Weekly", "Monthly", "Daily"],
        "Policy Type": ["Basic", "Comprehensive", "Premium"]
    }
    for column, categories in mapping_definitions.items():
        mapping = {category: index for index, category in enumerate(categories)}
        df[column]=df[column].replace(mapping).astype(int)

    print("Encoding")
    label_encoders = {}
    for col in ['Gender', 'Marital Status','Occupation','Location','Smoking Status','Property Type']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future use

    print("outlier removal")

    # Handle outliers via capping
    outlier_features = ['Annual Income', 'Premium Amount']
    for feature in outlier_features:
        upper_cap = df[feature].quantile(0.99)
        lower_cap = df[feature].quantile(0.01)
        df[feature] = df[feature].clip(upper=upper_cap, lower=lower_cap)

    print("transforamtion")

    # Apply Box-Cox transformation
    df['Annual Income'], income_lambda = stats.boxcox(df['Annual Income'] + 1)
    df['Premium Amount'], premium_lambda = stats.boxcox(df['Premium Amount'] + 1)

    print("scaling")
    # Scale numerical features
    scaler = StandardScaler()
    df['Annual Income'] = scaler.fit_transform(df[['Annual Income']])

    if train:
        return df, scaler, label_encoders, income_lambda, premium_lambda
    return df

# Load data

data, scaler1, encoders1, income_lambda1, premium_lambda1 = preprocess_data(data)

print("pre-process done")

# Feature Selection (Same as Training)
X = data.drop(columns=["Premium Amount"])
y = data["Premium Amount"]

# Predict on Training Data
y_train_pred = best_model.predict(X)

print("prediction done")

# Inverse Box-Cox Transformations
y_train_pred_inv = inv_boxcox(y_train_pred, premium_lambda)
y_train_actual_inv = inv_boxcox(y, premium_lambda)

print("inversing boxcox")

# Compute Scale Factor
scale_factor = np.mean(y_train_actual_inv / y_train_pred_inv)

print("scale factor",scale_factor)

# Save Scale Factor
pickle.dump(scale_factor, open("E:/AI engineer/Guvi/Capstone Projects/Project3/fresh_clone/SmartPremium/models/scale_factor.pkl", "wb"))
print(f"✅ Scale factor saved successfully: {scale_factor:.4f}")

# Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_train_actual_inv, y_train_pred_inv ))
mae = mean_absolute_error(y_train_actual_inv, y_train_pred_inv )
r2 = r2_score(y_train_actual_inv, y_train_pred_inv )

print(f"📊 Model Performance After Scaling:")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 R2 Score: {r2:.4f}")

