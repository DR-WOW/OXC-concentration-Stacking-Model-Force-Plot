import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt

# Ensure st.set_page_config is called at the beginning of the script
st.set_page_config(layout="wide", page_title="Stacking Model Prediction and SHAP Visualization", page_icon="📊")

# Import custom classes
from sklearn.base import RegressorMixin, BaseEstimator
from pytorch_tabnet.tab_model import TabNetRegressor

# Define the TabNetRegressorWrapper class
class TabNetRegressorWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
    
    def fit(self, X, y, **kwargs):
        # Convert X to a NumPy array
        X = X.values if isinstance(X, pd.DataFrame) else X
        # Convert y to a NumPy array and ensure it is two-dimensional
        y = y.values if isinstance(y, pd.Series) else y
        y = y.reshape(-1, 1)  # Ensure y is two-dimensional
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        # Convert X to a NumPy array
        X = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X, **kwargs).flatten()  # Flatten the prediction result to a one-dimensional array

# Load the model
model_path = "stacking_regressor_model.pkl"
try:
    stacking_regressor = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    raise

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

# Input for the true value
true_value = st.sidebar.number_input("True value (for accuracy comparison) (mg/L)", min_value=0.0, value=0.0)

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

        # Accuracy display logic
        if true_value > 0 and prediction is not None:
            st.header("📊 Prediction Accuracy")
            st.write("Display the model's absolute and relative accuracy.")

            # Calculate absolute and relative accuracy
            absolute_accuracy = abs(prediction - true_value)
            relative_accuracy = abs((prediction - true_value) / true_value) * 100 if true_value != 0 else 0

            # Display accuracy metrics
            st.subheader("Accuracy Metrics")
            st.write(f"Absolute Accuracy: {absolute_accuracy:.2f} mg/L")
            st.write(f"Relative Accuracy: {relative_accuracy:.2f}%")

            # Plot scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(true_value, prediction, alpha=0.5, color='blue', label='Prediction')
            ax.plot([0, max(true_value, prediction)], [0, max(true_value, prediction)], color='red', linestyle='--', label='Ideal Line')
            ax.set_xlabel('True Values (mg/L)')
            ax.set_ylabel('Predicted Values (mg/L)')
            ax.set_title('Prediction Accuracy')
            ax.legend()

            # Add metrics information
            textstr = '\n'.join((
                f'Absolute Accuracy: {absolute_accuracy:.2f} mg/L',
                f'Relative Accuracy: {relative_accuracy:.2f}%'))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

            # Display the plot
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Visualization display
st.header("SHAP Visualization Analysis")
st.write("""
The following charts display the model's SHAP analysis results, including the feature contributions of the first-layer base learners, the second-layer meta-learner, and the overall Stacking model.
""")

# SHAP visualization for the first-layer base learners
st.subheader("1. First-layer Base Learners")
st.write("Feature contribution analysis of the base learners (GBDT, XGBoost, LightGBM, CatBoost, TabNet, LASSO, etc.)")
first_layer_img = "SHAP Feature Importance of Base Learners in the First Layer of Stacking Model.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="SHAP contribution analysis of the first-layer base learners", use_column_width=True)
except FileNotFoundError:
    st.warning("SHAP image file for the first-layer base learners not found.")

# SHAP visualization for the second-layer meta-learner
st.subheader("2. Second-layer Meta-Learner")
st.write("Feature contribution analysis of the meta-learner (Linear Regression)")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="SHAP contribution analysis of the second-layer meta-learner", use_column_width=True)
except FileNotFoundError:
    st.warning("SHAP image file for the second-layer meta-learner not found.")

# SHAP visualization for the overall Stacking model
st.subheader("3. Overall Stacking Model")
st.write("Feature contribution analysis of the overall Stacking model")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="SHAP contribution analysis of the overall Stacking model", use_column_width=True)
except FileNotFoundError:
    st.warning("SHAP image file for the overall Stacking model not found.")

# Footer
st.markdown("---")
st.header("Summary")
st.write("""
Through this page, you can:
1. Perform real-time predictions using input feature values.
2. Gain an intuitive understanding of the feature contributions of the first-layer base learners, the second-layer meta-learner, and the overall Stacking model.
3. Compare the predicted value against the true value and evaluate the accuracy of the prediction.
These analyses help to deeply understand the model's prediction logic and the importance of features.
""")
