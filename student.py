import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def preprocess_data(df):
    """Preprocesses the input DataFrame."""
    df_cleaned = df.copy()
    imputer = SimpleImputer(strategy="mean")
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]])
    actual_categorical_column = 'GRADE' #Replace with your actual column name
    df_cleaned[actual_categorical_column] = df_cleaned[actual_categorical_column].astype('category')
    return df_cleaned


def engineer_features(df):
    """Engineers features for the DataFrame."""
    df_cleaned = df.copy()
    scaler = StandardScaler()
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    scaled_features = scaler.fit_transform(df_cleaned[numeric_columns])
    df_cleaned[numeric_columns] = scaled_features
    actual_feature_column = 'score' #Replace with your actual column name
    if actual_feature_column in df_cleaned.columns:
        df_cleaned['feature_squared'] = df_cleaned[actual_feature_column] ** 2
    else:
        st.error(f"Error: Column '{actual_feature_column}' not found!")
    return df_cleaned


def train_models(X, y):
    """Trains the machine learning models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models, X_test, y_test


def evaluate_models(models, X_test, y_test):
    """Evaluates the trained models."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
    return results


def encode_categorical(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X, y


st.title("Student Performance Prediction App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    df = engineer_features(df)
    actual_target_column_name = 'GRADE' #Replace with your actual column name
    if actual_target_column_name in df.columns:
        X, y = encode_categorical(df, actual_target_column_name)
        models, X_test, y_test = train_models(X, y)
        results = evaluate_models(models, X_test, y_test)
        for name, result in results.items():
            st.subheader(name)
            st.write("Classification Report:\n", result['classification_report'])
            st.write("Confusion Matrix:\n", result['confusion_matrix'])
            st.write("Accuracy:", result['accuracy'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix for {name}")
            st.pyplot(plt)
    else:
        st.error(f"Error: Target column '{actual_target_column_name}' not found in the dataset.")
