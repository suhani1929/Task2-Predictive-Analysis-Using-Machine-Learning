# Task2-Predictive-Analysis-Using-Machine-Learning
# 📊Telco Customer Churn & Billing

This project showcases a machine learning-based predictive analysis using the Telco Customer Churn dataset. The objective is to build two separate models using supervised learning techniques — one for **classification** (to predict customer churn) and the other for **regression** (to predict monthly billing charges). The goal is to derive actionable insights that can help businesses in customer retention and revenue prediction.

---

## 🎯 Objectives

1. **Classification Task**: Predict whether a customer is likely to churn based on attributes such as tenure, services subscribed, contract type, and payment method.
2. **Regression Task**: Estimate the `MonthlyCharges` a customer is expected to pay using various demographic and service-related features.

This two-pronged approach helps both in **customer retention strategy** and **revenue forecasting**.

---

## 📁 Dataset Description

- **Name**: Telco Customer Churn Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: ~7,000 customer entries
- **Key Features**:
  - Demographic: Gender, SeniorCitizen, Partner, Dependents
  - Service Info: InternetService, OnlineSecurity, StreamingTV, TechSupport
  - Billing Info: MonthlyCharges, TotalCharges, PaymentMethod
  - Target variables: `Churn` (Classification), `MonthlyCharges` (Regression)

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing
- Removed missing or invalid records.
- Converted `TotalCharges` to numerical.
- Encoded categorical variables using `LabelEncoder` and `pd.get_dummies()`.
- Applied `StandardScaler` for normalization (regression only).

### 2. Feature Selection
- **Classification**: Used `SelectKBest` with Chi-Square scoring to select top predictors for churn.
- **Regression**: Applied `f_regression` to identify features most correlated with `MonthlyCharges`.

### 3. Model Building
- **Classification Model**: `RandomForestClassifier`
- **Regression Model**: `RandomForestRegressor`

---

## 🧪 Evaluation Metrics

### 🟢 Classification (Churn)
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- Visualization: Confusion Matrix plot saved as `Confusion_Matrix_Churn_Classification.png`

### 🔵 Regression (MonthlyCharges)
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**
- Visualization: Actual vs. Predicted plot saved as `Actual_vs_Predicted_MonthlyCharges.png`

---

## 📊 Visualizations

- Confusion Matrix: Shows model's ability to correctly classify churn vs. non-churn.
- Actual vs. Predicted Plot: Evaluates how close predictions are to actual billing amounts.

Both figures help communicate the performance of the models visually, making it easier to interpret for non-technical stakeholders.

---

## 🔧 Technologies & Libraries Used

- **Python 3.10+**
- **VS Code**
- **Libraries**:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

---

## 📁 File Structure

```plaintext
├── Telco_Churn_Classification_Model.py
├── Telco_MonthlyCharges_Regression_Model.py
├── figures/
│   ├── Confusion_Matrix_Churn_Classification.png
│   └── Actual_vs_Predicted_Monthlycharges.png
|   └── Classification_Outcome.png
|   └── Regression_Outcome.png
├── dataset/
│   └── Telco-Customer-Churn.csv
└── README.md

---

## 🔍 Insights & Conclusion
- Customers with month-to-month contracts and no technical support are more likely to churn.

- Billing amount correlates strongly with internet services, support options, and tenure.

- Classification model achieved high accuracy, demonstrating that customer behavior is predictable with the right features.

- Regression model showed good fit, though variability may increase with inconsistent billing options or discounts.

These insights are valuable for customer retention campaigns, upselling strategies, and forecasting departmental revenue.
