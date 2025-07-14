import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved models and scaler
MODEL_PATH = 'voting_classifier_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Load the ensemble model and scaler
voting_model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))

# Title and description
st.title("Heart Attack Prediction with Voting Classifier")
st.write("""
This application predicts the likelihood of a person getting a heart attack using a Voting Classifier.
""")

# Sidebar: Input Parameters
st.sidebar.header("Input Parameters")

def user_input_features():
    # Input the feature data via Streamlit widgets
    age = st.sidebar.slider("Age", 18, 100, 25)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 500, 200)
    blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 80, 200, 120)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 40, 120, 70)
    diabetes = st.sidebar.selectbox("Diabetes (Yes/No)", ["No", "Yes"])
    family_history = st.sidebar.selectbox("Family History of Disease (Yes/No)", ["No", "Yes"])
    smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
    obesity = st.sidebar.selectbox("Obesity (Yes/No)", ["No", "Yes"])
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption (Yes/No)", ["No", "Yes"])
    exercise_hours = st.sidebar.slider("Exercise Hours Per Week", 0.00, 20.00, 3.00, 0.01)
    diet = st.sidebar.selectbox("Diet", ["Unhealthy", "Average", "Healthy"])
    previous_heart_problems = st.sidebar.selectbox("Previous Heart Problems (Yes/No)", ["No", "Yes"])
    medication_use = st.sidebar.selectbox("Medication Use (Yes/No)", ["No", "Yes"])
    stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    sedentary_hours = st.sidebar.slider("Sedentary Hours Per Day", 0.00, 24.00, 8.00, 0.01)
    bmi = st.sidebar.slider("BMI (kg/m²)", 10.00, 50.00, 25.00, 0.01)
    triglycerides = st.sidebar.slider("Triglycerides (mg/dL)", 50, 1500, 150)
    physical_activity_days = st.sidebar.slider("Physical Activity Days Per Week", 0, 7, 3)
    sleep_hours = st.sidebar.slider("Sleep Hours Per Day", 4, 12, 8)

    # Encode categorical variables (the same encoding as training)
    sex = 1 if sex == "Male" else 0
    family_history = 1 if family_history == "Yes" else 0
    smoking = {"No": 0, "Yes": 1}[smoking]
    obesity = 1 if obesity == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    alcohol_consumption = {"No": 0, "Yes": 1}[alcohol_consumption]
    diet = {"Unhealthy": 0, "Average": 1, "Healthy": 2}[diet]
    previous_heart_problems = 1 if previous_heart_problems == "Yes" else 0
    medication_use = 1 if medication_use == "Yes" else 0

    # Create a DataFrame for the user input
    data = {
        "Age": age,
        "Sex": sex,
        "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Diabetes": diabetes,
        "Family History": family_history,
        "Smoking": smoking,
        "Obesity": obesity,
        "Alcohol Consumption": alcohol_consumption,
        "Exercise Hours Per Week": exercise_hours,
        "Diet": diet,
        "Previous Heart Problems": previous_heart_problems,
        "Medication Use": medication_use,
        "Stress Level": stress_level,
        "Sedentary Hours Per Day": sedentary_hours,
        "BMI": bmi,
        "Triglycerides": triglycerides,
        "Physical Activity Days Per Week": physical_activity_days,
        "Sleep Hours Per Day": sleep_hours
    }

    return pd.DataFrame(data, index=[0])

# Collect user input
df = user_input_features()

# Display user input
st.subheader("Input Parameters")
st.write(df)

# Make prediction
if st.button("Predict"):
    # Standardize the user input data (apply the same transformation as in training)
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = voting_model.predict(df_scaled)
    probability = voting_model.predict_proba(df_scaled)

    # Define likelihood classes
    likelihood_classes = ["Not Likely", "Likely"]
    result = likelihood_classes[int(prediction[0])]

    # Format probabilities to standard floats and round to 2 decimal places
    formatted_probabilities = [round(float(prob), 2) for prob in probability[0]]

    # Show prediction result
    st.subheader("Prediction Result")
    st.write(f"The likelihood of a heart attack is: **{result}**")
    
    # Show prediction probabilities
    st.subheader("Prediction Probabilities")
    st.write(f"Probabilities: {formatted_probabilities}")

