# Telco Customer Churn Prediction

## Overview
This project aims to analyze customer churn for a telecom company using machine learning models. The dataset contains various customer attributes such as demographics, services, and billing information. The goal is to identify key factors influencing churn and develop a predictive model using **H2O.ai AutoML**.

## Dataset
The dataset contains the following key features:
- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Information:** Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure
- **Target Variable:** Churn (Yes/No)

## Workflow
1. **Data Preprocessing**
   - Handle missing values
   - Convert categorical variables into numerical format (Label Encoding, One-Hot Encoding)
   - Normalize continuous variables (Tenure, MonthlyCharges, TotalCharges)

2. **Exploratory Data Analysis (EDA)**
   - Check class distribution
   - Visualize correlation heatmaps
   - Identify multicollinearity using Variance Inflation Factor (VIF)

3. **Feature Engineering & Selection**
   - Remove highly correlated variables
   - Evaluate statistical significance of features

4. **Model Building using H2O AutoML**
   - Train multiple machine learning models (GBM, XGBoost, Deep Learning, Random Forest, Logistic Regression)
   - Automatically optimize hyperparameters
   - Select the best-performing model based on AUC, LogLoss, and accuracy

5. **Evaluation**
   - Confusion Matrix
   - ROC Curve & AUC Score
   - Feature Importance Analysis

## Dependencies
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- H2O.ai AutoML
- etc....
