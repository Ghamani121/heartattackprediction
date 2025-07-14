# Heart Attack Prediction using Ensemble Machine Learning

This project is an intelligent, interactive web application that predicts the likelihood of a heart attack (myocardial infarction) using ensemble machine learning techniques. Built with Python and deployed using Streamlit, the app is designed to aid early diagnosis in a fun, educational, and research-oriented context.

---

## Technologies Used

- **Python 3.9+**
- **Scikit-learn**: For machine learning model building
- **XGBoost**: High-performance boosting classifier
- **Imbalanced-learn (SMOTE)**: Class imbalance correction
- **Pandas, NumPy, Matplotlib, Seaborn**: Data handling & visualization
- **Joblib**: Model serialization
- **Streamlit**: Interactive UI for real-time predictions

---

## Architecture

The system is divided into the following core components:

- **Data Preprocessing (`fe.py`)**  
  Handles loading and transforming the data, performing scaling, encoding, SMOTE, etc.

- **Model Training (`ensemble.py`)**  
  Trains Logistic Regression, Random Forest, XGBoost, and combines them using a Voting Classifier. Exports `.pkl` models for deployment.

- **Streamlit App (`fe.py`)**  
  Interactive UI where users input values, choose a classifier, and see the prediction in real time.

- **Model Files (`scaler.pkl`, `voting_classifier_model.pkl`)**  
  Saved using Joblib for reuse in deployment.

---

## Features

- Upload and preprocess patient data
- Predict heart attack risk using:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Voting Classifier (ensemble)
- Balance class distribution using SMOTE
- Visualize confusion matrix and performance
- Streamlit-based GUI for seamless predictions

---

## Model Performance

| Classifier         | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 87.37%   | 0.87      | 0.87   | 0.87     |
| Random Forest      | 99.32%   | 0.99      | 0.99   | 0.99     |
| XGBoost            | 99.54%   | 1.00      | 1.00   | 1.00     |
| **Voting Classifier** | **99.39%** | **0.99** | **0.99** | **0.99** |

---

## Local Setup Instructions

### Prerequisites:
- Python 3.9+
- Git (optional)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/heart-attack-prediction.git
cd heart-attack-prediction

# 2. Set up a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install required packages
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost streamlit joblib

# 4. (Optional) Update dataset path in ensemble.py if needed

# 5. Train the model (optional, if .pkl already exists)
python ensemble.py

# 6. Launch the Streamlit app
streamlit run fe.py

```
---

## Screenshots

<img width="1911" height="815" alt="image" src="https://github.com/user-attachments/assets/c2ef12ea-61b5-4e5e-9fdc-e69504fc918a" />

