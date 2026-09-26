# 🩺 Diabetes Prediction System

An end-to-end Machine Learning project that predicts the likelihood of a patient developing diabetes based on clinical metrics. 

This project includes data preprocessing, exploratory data analysis, model training using Logistic Regression, and a real-time interactive web application deployed via Streamlit.

## 🚀 Project Overview
* **Dataset:** PIMA Indians Diabetes Database (768 records, 8 clinical features).
* **Machine Learning Algorithm:** Logistic Regression.
* **Accuracy:** Achieved 77.22% Cross-Validation Accuracy.
* **Frontend:** Streamlit web dashboard for real-time predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit, Pickle

## 📊 Key Features
1. **Data Preprocessing:** Handled biologically invalid '0' values (e.g., BMI, Blood Pressure) via median imputation.
2. **Feature Scaling:** Applied `StandardScaler` to normalize feature distributions.
3. **Model Evaluation:** Utilized 5-Fold Cross Validation and Confusion Matrices to ensure model stability and minimize false negatives.
4. **Interactive UI:** A user-friendly dashboard allowing non-technical users to input physiological parameters and receive instant diagnostic predictions with confidence scores.

## 💻 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/Diabetes-Prediction-Machine-Learning.git](https://github.com/YourUsername/Diabetes-Prediction-Machine-Learning.git)
cd Diabetes-Prediction-Machine-Learning