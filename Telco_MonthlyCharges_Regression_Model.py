# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.head()

# Data Cleaning
df.drop('customerID', axis=1, inplace=True) 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encoding categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = le.fit_transform(df[column])

# Regression Task - Predicting Monthly Charges
x_reg = df.drop('MonthlyCharges', axis=1)
y_reg = df['MonthlyCharges']

# Feature Selection
selector_reg = SelectKBest(score_func=f_regression, k=10)
x_reg_selected = selector_reg.fit_transform(x_reg, y_reg)
selected_features_reg = x_reg.columns[selector_reg.get_support()]
print("Selected features for regression:\n", selected_features_reg.tolist())

# Train-Test Split
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(df[selected_features_reg], y_reg, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
x_train_reg = scaler.fit_transform(x_train_reg)
x_test_reg = scaler.transform(x_test_reg)

# Train Model
reg = RandomForestRegressor()
reg.fit(x_train_reg, y_train_reg)

# Predict
y_pred_reg = reg.predict(x_test_reg)

# Evaluate
print('\n Regression Evaluation:')
print('\n Regression Mean Absolute Error:', mean_absolute_error(y_test_reg, y_pred_reg))
print('\n Regression Mean Squared Error:', mean_squared_error(y_test_reg, y_pred_reg))
print('\n Regression Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
print('\n Regression R2 Score:', r2_score(y_test_reg, y_pred_reg))

# Plotting Actual vs Predicted Monthly Charges
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, color='teal')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], '--', color='gray', label='Perfect Prediction')
plt.title('Actual vs Predicted Monthly Charges')
plt.xlabel('Actual Monthly Charges')
plt.ylabel('Predicted Monthly Charges')
plt.legend()
plt.grid(True)
plt.show()