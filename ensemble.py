import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from xgboost import XGBClassifier  # Import XGBoost classifier
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # Import SMOTE

def train_and_save_model():
    # Load data from external CSV file for training
    df = pd.read_csv(r"C:\Users\Ghamani\Downloads\miniproject\ia1implementation\heart_attack_prediction_dataset.csv")
    
    # Drop irrelevant columns
    df = df.drop(columns=['Patient ID', 'Income', 'Country', 'Continent', 'Hemisphere'])
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Separate features (X) and target variable (y)
    X = df.drop('Heart Attack Risk', axis=1)  # Features
    y = df['Heart Attack Risk']  # Target

    # Outlier handling using IQR
    for col in X.columns:
        Q1, Q3 = np.percentile(X['Sedentary Hours Per Day'], [25, 75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X['Sedentary Hours Per Day'] = np.where(X['Sedentary Hours Per Day'] > upper, upper,\
                                               np.where(X['Sedentary Hours Per Day'] < lower, lower, X['Sedentary Hours Per Day']))

    # Apply Label Encoding to categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        label_encoder.fit(X[col])
        X[col] = label_encoder.transform(X[col])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data to address class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Define classifiers
    rf_clf = RandomForestClassifier(random_state=42)
    logreg_clf = LogisticRegression(random_state=42)
    xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss')  # XGBoost

    # Define the Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('logreg', logreg_clf), ('xgb', xgb_clf)],  # Added XGBoost
        voting='soft'
    )

    # Train the individual models and the Voting Classifier
    rf_clf.fit(X_train_smote, y_train_smote)
    logreg_clf.fit(X_train_smote, y_train_smote)
    xgb_clf.fit(X_train_smote, y_train_smote)
    voting_clf.fit(X_train_smote, y_train_smote)

    # Save the trained models and scaler
    with open('voting_classifier_model.pkl', 'wb') as f:
        pickle.dump(voting_clf, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Evaluate the individual models
    rf_pred = rf_clf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:}")
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    print("Random Forest Confusion Matrix:")
    rf_cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title("Random Forest Confusion Matrix")
    plt.show()

    logreg_pred = logreg_clf.predict(X_test_scaled)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    print(f"Logistic Regression Accuracy: {logreg_accuracy:}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logreg_pred))
    print("Logistic Regression Confusion Matrix:")
    logreg_cm = confusion_matrix(y_test, logreg_pred)
    sns.heatmap(logreg_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    xgb_pred = xgb_clf.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred))
    print("XGBoost Confusion Matrix:")
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title("XGBoost Confusion Matrix")
    plt.show()

    # Evaluate the Voting Classifier
    voting_pred = voting_clf.predict(X_test_scaled)
    voting_accuracy = accuracy_score(y_test, voting_pred)
    print(f"Voting Classifier Accuracy: {voting_accuracy:}")
    print("Voting Classifier Classification Report:")
    print(classification_report(y_test, voting_pred))
    print("Voting Classifier Confusion Matrix:")
    voting_cm = confusion_matrix(y_test, voting_pred)
    sns.heatmap(voting_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title("Voting Classifier Confusion Matrix")
    plt.show()

# Train the model (You should run this once, then comment it out for production)
train_and_save_model()
