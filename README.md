# **SmartPremium - Insurance Premium Prediction**  

## **📌 Project Overview**
SmartPremium is an **AI-driven Insurance Premium Prediction System** that utilizes **machine learning models** to estimate insurance premiums based on customer details. The project implements advanced **data preprocessing, feature engineering, and model optimization** techniques to improve prediction accuracy.  

It is built using **Python, Scikit-Learn, XGBoost, Bayesian Optimization, MLflow, and Streamlit** for an **end-to-end** machine learning pipeline.  

---

## **🛠 Features**
✅ **Data Preprocessing & Feature Engineering:**  
- Handling missing values  
- Encoding categorical variables  
- Applying **Box-Cox transformation** for skewed features  
- **Feature scaling** using StandardScaler  
- **Feature binning** for numerical attributes  

✅ **Machine Learning Models Implemented:**  
- **Linear Regression**  
- **Decision Tree Regressor**  
- **Random Forest Regressor** (Tuned with Bayesian Optimization)  
- **XGBoost Regressor** (Tuned with Bayesian Optimization)  

✅ **Model Evaluation Metrics:**  
- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score (Coefficient of Determination)**  
- **RMSLE (Root Mean Squared Logarithmic Error)**  

✅ **MLflow Integration for Model Tracking**  
- Logs **hyperparameters, metrics, and trained models**  
- Enables **comparison of different models**  
- Helps in **versioning and reproducibility**  

✅ **Streamlit UI for Predictions**  
- Enter **customer details** to predict **insurance premium**  
- Upload a **CSV file** with customer data to get **bulk predictions**  

---

## **🚀 Getting Started**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Mahaa-Lakshmi/SmartPremium.git
cd SmartPremium
```

### **2️⃣ Create & Activate Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate
# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
### **4️⃣ Train & Track Model Performance with MLflow**
Run the training script to preprocess data, train models, and log them in MLflow:
```bash
pip install -r requirements.txt
```
This below script will:

Load and preprocess training data.Train multiple models. Log model parameters & performance in MLflow.

Save the best-performing model
```bash
python Scripts\pipeline1.py
```

### **5️⃣ View MLflow Model Parameters & Metrics**
After training, you can open MLflow UI to explore logged models, parameters, and metrics. Run this command:
```bash
mlflow ui --backend-store-uri "file:///E:\AI engineer\Guvi\Capstone Projects\Project3\fresh_clone\SmartPremium\mlruns\kaggle\temp\mlruns"
```
Open http://127.0.0.1:5000 in your browser to explore model runs.

You can compare RMSE, MAE, R² Score for different models.

### **6️⃣ Run the Streamlit UI**
To interact with the model via a web app:
```bash
streamlit run app.py
```
You can enter individual customer details to get a premium estimate.

(or)

You can upload a CSV file to get bulk predictions.

## **📂 Project Structure**
```graphql
SmartPremium/
│── mlruns/                          # MLflow logs & model tracking
│── models/                          # Trained models & preprocessors
│   ├── best_model.pkl               # Best-performing model
│   ├── scaler.pkl                    # StandardScaler for feature scaling
│   ├── label_encoders.pkl            # Encoders for categorical variables
│   ├── boxcox_lambdas.pkl            # Box-Cox transformation parameters
│   ├── Linear Regression.pkl         # Linear Regression model 
│   ├── Decision Tree.pkl             # Decision Tree model
│   ├── Random Forest.pkl            # Random Forest model
│   ├── XGBoost.pkl                   # XGBoost model
│── data/                            # Dataset folder
│   ├── train.csv                     # Training dataset
│   ├── test.csv                      # Test dataset
│── scripts/                         # Python scripts for training & analysis
│   ├── pipeline.py                  # Data preprocessing  and model tuning and buildingfunctions
│   ├── app.py                      # Streamlit UI for predictions
│── requirements.txt                  # Required Python packages
│── README.md                         # Documentation
```
## **📌 Model Training & Optimization**
### **1️⃣ Data Preprocessing**
- Handle missing values
- Feature binning (Grouping similar values into categories)
- Encoding categorical variables
- Box-Cox transformation for normalizing skewed features
- StandardScaler for numerical features

### **2️⃣ Model Training**
- Hyperparameter tuning using Bayesian Optimization
- Performance evaluation using RMSE, MAE, R² Score, RMSLE

### **3️⃣ Model Selection & Saving**
- The best-performing model is saved as best_model.pkl

## **📌 How to Test Model on New Data?**
### **1️⃣ Single Prediction**
- Run Streamlit UI and enter customer details manually to get premium predictions.

### **2️⃣ Bulk Predictions using CSV Upload**
- Prepare a CSV file (same format as test.csv)
- Run Streamlit UI and upload the CSV file
- The app will process the file and generate a downloadable results file

## **⚙️ Dependencies**
```pandas 
numpy==1.26.4
matplotlib 
seaborn 
scikit-learn==1.2.2
xgboost  
streamlit
nbformat
mlflow
scikit-optimize 
bayesian-optimization
```

## **🔹 Notes**
- The project automatically selects the best model based on RMSE.
- MLflow allows tracking of multiple experiments and comparing models.
- Streamlit UI makes it easy to use for non-technical users.

## **📌 Future Improvements**
- Improve feature engineering for better accuracy
- Implement deep learning models for better performance
- Deploy model using Flask/FastAPI for API-based predictions
- Add AutoML techniques for hyperparameter tuning

## **💡 Conclusion**
This project demonstrates an end-to-end machine learning workflow for insurance premium prediction. It preprocesses data, trains models, optimizes performance, and provides a user-friendly UI for predictions.