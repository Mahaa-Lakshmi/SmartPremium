import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pickle
from mlflow.models import infer_signature

from scipy import stats
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings("ignore")


# Initialize MLflow
mlflow.set_tracking_uri("file:///kaggle/temp/mlruns")
mlflow.set_experiment("Insurance Premium Prediction")

# ------------------------ Data Preprocessing ------------------------

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
data = pd.read_csv("/kaggle/input/files-smartpremium/train.csv")
data, scaler, encoders, income_lambda, premium_lambda = preprocess_data(data)

# Save preprocess objects
pickle.dump(scaler, open("/kaggle/working/scaler.pkl", "wb"))
pickle.dump(encoders, open("/kaggle/working/label_encoders.pkl", "wb"))
pickle.dump((income_lambda, premium_lambda), open("/kaggle/working/boxcox_lambdas.pkl", "wb"))

print("Saved 3 pre-process pkl")

# ------------------------ Feature Splitting ------------------------

# Define Features & Target
X = data.drop(columns=["Premium Amount"])
y = data["Premium Amount"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# ------------------------ Model Evaluation Function ------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """ Train, evaluate, and return model metrics. """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Reverse Box-Cox transformation
    y_test_exp = (y_test * premium_lambda) ** (1 / premium_lambda)
    y_pred_exp = (y_pred * premium_lambda) ** (1 / premium_lambda)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    r2 = r2_score(y_test_exp, y_pred_exp)
    rmsle = np.sqrt(mean_squared_log_error(y_test_exp, np.abs(y_pred_exp)))

    print(f"{model} rmse:{rmse} mae:{mae} r2:{r2} rmsle:{rmsle}")

    return rmse, mae, r2, rmsle

# ------------------------ Model Tuning Functions ------------------------
"""
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

important_features = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top_features = important_features.head(10).index  # Select top 10 most important features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
"""

def tune_random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    """ Bayesian Optimization function for Random Forest. """
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        max_samples=0.8
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return -np.sqrt(mean_squared_error((y_test * premium_lambda) ** (1 / premium_lambda), (y_pred * premium_lambda) ** (1 / premium_lambda)))

# Define Bayesian Optimization Function
def tune_xgboost(n_estimators, learning_rate, max_depth, colsample_bytree, subsample):
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        early_stopping_rounds=10  # Stop early if no improvement
    )

    # Fit model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predict on test set
    y_pred = model.predict(X_test)


    return -np.sqrt(mean_squared_error((y_test * premium_lambda) ** (1 / premium_lambda), (y_pred * premium_lambda) ** (1 / premium_lambda)))

# ------------------------ Train, Tune & Save Best Model ------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_split=13, random_state=42),
    "XGBoost": XGBRegressor(),
    "Random Forest": RandomForestRegressor()
  }  

print("tuning XGBoost started")

# Tune XGBoost
xgb_optimizer = BayesianOptimization(f=tune_xgboost, pbounds={
    "n_estimators": (50, 200),
    "learning_rate": (0.01, 0.3),
    "max_depth": (3, 10),
    "colsample_bytree": (0.6, 1.0),
    "subsample": (0.6, 1.0)
}, random_state=42)

xgb_optimizer.maximize(init_points=5, n_iter=10)

best_params = xgb_optimizer.max["params"]
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])

models["XGBoost tuned"] = XGBRegressor(**best_params, random_state=42, n_jobs=-1)

print("tuning XGBoost done")

print("tuning RF started")

# Tune Random Forest
rf_optimizer = BayesianOptimization(f=tune_random_forest, pbounds={
        "n_estimators": (50, 300),       # Integer range
        "max_depth": (5, 20),            # Ensure whole numbers
        "min_samples_split": (2, 20),    # Integer range
        "min_samples_leaf": (1, 10),     # Integer range
        "max_features": (0.5, 1.0)       # This can remain float
}, random_state=42, allow_duplicate_points=True)

rf_optimizer.maximize(init_points=5, n_iter=10)

best_params = rf_optimizer.max["params"]
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])
best_params["min_samples_split"] = int(best_params["min_samples_split"])
best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

models["Random Forest"] = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)


print("tuning RF done")

# MLflow Logging
best_model, best_rmse,best_model_name = None, float("inf"),""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(name,"mlflowing")
        rmse, mae, r2, rmsle = evaluate_model(model, X_train, y_train, X_test, y_test)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2, "RMSLE": rmsle})

        model_path = f"/kaggle/working/{name}.pkl"
        pickle.dump(model, open(model_path, "wb"))
        #mlflow.sklearn.log_model(model, name,input_example=X_train.iloc[:1],signature=infer_signature(X_train, model.predict(X_train)))
        print(name, "model saved locally & MLflow metrics logged!")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name=name
            mlflow.sklearn.log_model(best_model, "best_model")

# Save best model
pickle.dump(best_model, open("/kaggle/working/best_model.pkl", "wb"))
print("✅ Best model saved!",best_model_name)