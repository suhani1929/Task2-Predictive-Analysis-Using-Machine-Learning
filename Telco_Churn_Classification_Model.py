# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

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

# Classification Task - Predicting Churn
x_class = df.drop('Churn', axis=1)
y_class = df['Churn']

# Feature Selection (Chi-Square)
selector_class = SelectKBest(score_func=chi2, k=10)
selector_class.fit(x_class, y_class)

# Keep only the selected features
selected_features_class = x_class.columns[selector_class.get_support()]
print("Classification Selected Features:\n", selected_features_class.tolist())

x_class_selected = x_class[selected_features_class]  # Retain as DataFrame

# Train-Test Split
x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
    x_class_selected, y_class, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
x_train_class = scaler.fit_transform(x_train_class)
x_test_class = scaler.transform(x_test_class)

# Model Training
clf = RandomForestClassifier()
clf.fit(x_train_class, y_train_class)

# Prediction
y_pred_c = clf.predict(x_test_class)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test_class, y_pred_c))
print("\nClassification Report:\n", classification_report(y_test_class, y_pred_c))

sns.heatmap(confusion_matrix(y_test_class, y_pred_c), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Classification (Churn)")
plt.show()
