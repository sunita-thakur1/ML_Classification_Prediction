import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

st.set_page_config(page_title="ML Classification Prediction", layout="centered")
st.title("🔮 ML Classification Model Prediction")   
st.markdown("""
Welcome to the **Recommendation Model Predictor**! This app helps predict the best recommendation model for users based on their preferences and behavior.
### Key Features:
- **Interactive User Input**: Users can input personal details (e.g., age, cuisine preference, taste) to get a model recommendation.
- **Data Upload**: Option to upload a custom dataset.
- **Model Prediction**: A trained Random Forest Classifier predicts the best recommendation model based on the user's input.
- **Feature Importance**: Visual display of the top 10 most important features influencing the model's recommendations.
- **Simulated User Predictions**: Predictions for sample users are displayed to demonstrate the model's functionality.
- **Download Results**: Users can download simulated predictions in CSV format for further analysis.
""")

# Upload dataset
uploaded_file = st.file_uploader("📥 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Feature selection
all_columns = df.columns.tolist()
target_column = st.selectbox("🎯 Select Target Column", options=all_columns)
feature_columns = st.multiselect("🧬 Select Feature Columns", options=[col for col in all_columns if col != target_column])

if not feature_columns:
    st.warning("Please select at least one feature column.")
    st.stop()

@st.cache_resource
def train_model(df, features, target):
    X = df[features]
    y = df[target]

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder='passthrough')

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, acc, report

clf, accuracy, report = train_model(df, feature_columns, target_column)

st.write("### Preview of Data:")
st.write(df.head())
st.write("### Summary Statistics:")
st.write(df.describe())

# --- User Input Section ---
st.header("🧑 ML Model Prediction for a New User")

with st.form("user_form"):
    user_input = {}
    for col in feature_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            user_input[col] = st.selectbox(f"{col}", df[col].unique())
        elif np.issubdtype(df[col].dtype, np.integer):
            user_input[col] = st.number_input(f"{col}", value=int(df[col].mean()))
        elif np.issubdtype(df[col].dtype, np.floating):
            user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            user_input[col] = st.text_input(f"{col}")

    submitted = st.form_submit_button("Predict Model")

    if submitted:
        new_user = pd.DataFrame([user_input])
        prediction = clf.predict(new_user)[0]
        st.success(f"✅ Recommended Model: **{prediction}**")

# --- Feature Importance ---
with st.expander("📊 Show Feature Importances"):
    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["classifier"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx][:10][::-1], 
            [importances[i] for i in sorted_idx][:10][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

# --- Simulated Users ---
st.header("🧪 Simulated Users")
simulated_users = df[feature_columns].copy()
predicted_models = clf.predict(simulated_users)
simulated_users["Recommended_Model"] = predicted_models

st.dataframe(simulated_users)

# Optional: Download predictions
csv = simulated_users.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download All Predictions", csv, "all_predictions.csv", "text/csv")
