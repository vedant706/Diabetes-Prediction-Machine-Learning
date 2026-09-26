# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# Step 1: Data Collection
# ==========================================
print("Loading dataset...")
df = pd.read_csv('diabetes.csv')

# ==========================================
# Step 2: Data Preprocessing
# ==========================================
# Replacing zero values with NaN for specific columns where 0 is biologically impossible
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

# Fill NaN values with the median of the respective columns
for col in columns_to_replace:
    df[col].fillna(df[col].median(), inplace=True)

# Separate features (X) and target label (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale the features for better Logistic Regression performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# Step 3: Exploratory Data Analysis (EDA)
# ==========================================
# Note: In a script, plt.show() will pop up windows. 
# For a notebook, they will display inline.

# 1. Outcome Countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('Distribution of Diabetic vs Non-Diabetic Patients')
plt.savefig('outcome_distribution.png') # Save for project report

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# ==========================================
# Step 4: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ==========================================
# Step 5: Model Building (Logistic Regression)
# ==========================================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# ==========================================
# Step 6: Accuracy and Confusion Matrix
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')

# ==========================================
# Step 7: Cross Validation
# ==========================================
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=5)
print(f"\n--- Cross Validation ---")
print(f"CV Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# Save the model and scaler for the Streamlit frontend
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
print("\nModel and Scaler saved successfully for deployment!")