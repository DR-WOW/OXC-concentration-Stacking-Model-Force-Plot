import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt

# Ensure st.set_page_config is called at the beginning of the script
st.set_page_config(layout="wide", page_title="Stacking Model Prediction and SHAP Visualization", page_icon="📊")

# Load the model
model_path = "stacking_regressor_model.pkl"
try:
    stacking_regressor = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()
except EOFError:
    st.error("Model file is incomplete or corrupted. Please re-generate the model file.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load SHAP values
shap_values_path = "Final_stacking_shap_df3.xlsx"
try:
    stacking_shap_df3 = pd.read_excel(shap_values_path, index_col=0)
    st.success("SHAP values loaded successfully!")
except FileNotFoundError:
    st.error("SHAP values file not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load SHAP values: {e}")
    st.stop()

# Load test features and labels
test_features_path = "test_features.csv"
test_labels_path = "test_labels.csv"
try:
    test_features = pd.read_csv(test_features_path)
    test_labels = pd.read_csv(test_labels_path)
    st.success("Test data loaded successfully!")
except FileNotFoundError:
    st.error("Test data files not found. Please check the file paths.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load test data: {e}")
    st.stop()

# Set page title
st.title("📊 Stacking Model Prediction and SHAP Visualization")
st.write("""
By inputting feature values, you can obtain the model's prediction and understand the contribution of each feature using SHAP analysis.
""")

# Sidebar for feature input
st.sidebar.header("Feature Input Area")
st.sidebar.write("Please input feature values:")

# Define feature input ranges with units
SEX = st.sidebar.selectbox("Gender (1 = male, 0 = female)", [0, 1])
AGE = st.sidebar.number_input("Age (years)", min_value=0.1, max_value=18.0, value=5.0)
WT = st.sidebar.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=25.0)
Single_Dose = st.sidebar.number_input("Single dose per weight (mg/kg)", min_value=0.1, max_value=60.0, value=15.0)
Daily_Dose = st.sidebar.number_input("Daily dose (mg)", min_value=0.1, max_value=2400.0, value=450.0)
SCR = st.sidebar.number_input("Serum creatinine (μmol/L)", min_value=0.1, max_value=150.0, value=30.0)
CLCR = st.sidebar.number_input("Creatinine clearance rate (L/h)", min_value=0.1, max_value=200.0, value=90.0)
BUN = st.sidebar.number_input("Blood urea nitrogen (mmol/L)", min_value=0.1, max_value=50.0, value=5.0)
ALT = st.sidebar.number_input("Alanine aminotransferase (ALT) (U/L)", min_value=0.1, max_value=150.0, value=18.0)
AST = st.sidebar.number_input("Aspartate transaminase (AST) (U/L)", min_value=0.1, max_value=150.0, value=18.0)
CL = st.sidebar.number_input("Metabolic clearance of drugs (CL) (L/h)", min_value=0.1, max_value=100.0, value=3.85)
V = st.sidebar.number_input("Apparent volume of distribution (Vd) (L)", min_value=0.1, max_value=1000.0, value=10.0)

# Add prediction button
predict_button = st.sidebar.button("Predict")

# Main page for result display
if predict_button:
    st.header("Prediction Result (mg/L)")
    try:
        input_array = np.array([SEX, AGE, WT, Single_Dose, Daily_Dose, SCR, CLCR, BUN, ALT, AST, CL, V]).reshape(1, -1)
        prediction = stacking_regressor.predict(input_array)[0]
        
        # Ensure the prediction is positive
        if prediction <= 0:
            prediction = 0.1  # Set a small positive value if prediction is non-positive
        
        st.success(f"Prediction result: {prediction:.2f} mg/L")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Visualization display
st.header("SHAP Visualization Analysis")
st.write("""
The following charts display the model's SHAP analysis results, including the feature contributions of the first-layer base learners, the second-layer meta-learner, and the overall Stacking model.
""")

# SHAP visualization for the overall Stacking model
st.subheader("1. Overall Stacking Model SHAP Summary Plot")
try:
    explainer = shap.KernelExplainer(stacking_regressor.predict, test_features)
    shap_values = explainer.shap_values(test_features)
    shap.summary_plot(shap_values, test_features)
    st.pyplot()
except Exception as e:
    st.error(f"Failed to generate SHAP summary plot: {e}")

# SHAP visualization for a single sample using Force Plot
st.subheader("2. SHAP Force Plot for a Single Sample")
sample_index = st.slider("Select a sample index", 0, len(test_features) - 1, 0)
try:
    shap.force_plot(explainer.expected_value, shap_values[sample_index, :], test_features.iloc[sample_index, :])
    st.write("Note: Force Plot may not render properly in Streamlit. Open the HTML file generated by SHAP for full interactivity.")
except Exception as e:
    st.error(f"Failed to generate SHAP force plot: {e}")

# Prediction accuracy plot
st.subheader("3. Prediction Accuracy Plot")
try:
    y_pred = stacking_regressor.predict(test_features)
    y_true = test_labels.values.flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction Accuracy Plot")
    st.pyplot()
except Exception as e:
    st.error
