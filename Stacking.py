import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 定义 TabNetRegressorWrapper 类
class TabNetRegressorWrapper:
    def __init__(self, **kwargs):
        from pytorch_tabnet.tab_model import TabNetRegressor
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

# 加载模型
model_path = "stacking_regressor_model.pkl"
try:
    # 确保在加载模型时，TabNetRegressorWrapper 类已经定义
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    raise  # Re-raise the exception for debugging

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Concentration Prediction", page_icon="📊")
st.title("📊 Concentration Prediction and SHAP Visualization")
st.write("""
By inputting feature values, you can obtain the model's prediction and understand the contribution of each feature using SHAP analysis. 

If a true value is provided, the model's absolute and relative accuracy, as well as precision within ±30% and ±3, will also be displayed.
""")

# 特征输入区域
st.sidebar.header("Feature Input Area")
st.sidebar.write("Please input feature values:")

# 定义特征输入范围
feature_ranges = {
    "SEX": {"type": "categorical", "options": [0, 1], "default": 0, "description": "Gender (0 = Female, 1 = Male)"},
    "AGE": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0, "description": "Patient's age (in years)"},
    "WT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 25.0, "description": "Patient's weight (kg)"},
    "Single_Dose": {"type": "numerical", "min": 0.0, "max": 60.0, "default": 15.0, "description": "Single dose of the drug per weight (mg/kg)"},
    "Daily_Dose": {"type": "numerical", "min": 0.0, "max": 2400.0, "default": 450.0, "description": "Total daily dose of the drug (mg)"},
    "SCR": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 30.0, "description": "Serum creatinine level (μmol/L)"},
    "CLCR": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 90.0, "description": "Creatinine clearance rate (L/h)"},
    "BUN": {"type": "numerical", "min": 0.0, "max": 50.0, "default": 5.0, "description": "Blood urea nitrogen level (mmol/L)"},
    "ALT": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "Alanine aminotransferase level (U/L)"},
    "AST": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "Aspartate transaminase level (U/L)"},
    "CL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 3.85, "description": "Metabolic clearance rate of the drug (L/h)"},
    "V": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 10.0, "description": "Apparent volume of distribution of the drug (L)"}
}

# 动态生成输入界面
inputs = {}
for feature, config in feature_ranges.items():
    if config["type"] == "numerical":
        inputs[feature] = st.sidebar.number_input(
            f"{feature} ({config['description']})",
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"]
        )
    elif config["type"] == "categorical":
        inputs[feature] = st.sidebar.selectbox(
            f"{feature} ({config['description']})",
            options=config["options"],
            index=config["options"].index(config["default"])
        )

# 添加真实值输入框
true_value = st.sidebar.number_input("True Value (mg/L)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# 将输入特征转换为 Pandas DataFrame
features_df = pd.DataFrame([inputs])

# 如果模型在训练时使用了分类特征，确保这些特征是整数类型
cat_features = ["SEX"]  # 假设 SEX 是分类特征
features_df[cat_features] = features_df[cat_features].astype(int)

# 模型预测
prediction = None  # 初始化预测结果为 None
if st.button("Predict"):
    try:
        prediction = model.predict(features_df)[0]  # 预测结果是一个连续变量

        # 显示预测结果
        st.header("Prediction Result")
        st.success(f"Based on the feature values, the predicted concentration is {prediction:.2f} mg/L.")

        # 保存预测结果为图片
        fig, ax = plt.subplots(figsize=(8, 1))
        text = f"Predicted Concentration: {prediction:.2f} mg/L"
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes
        )
        ax.axis('off')
        plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
        st.image("prediction_text.png", use_column_width=True)

        # 可视化展示
        st.header("SHAP Visualization and Model Prediction Performance Analysis")
        st.write("""
        The following charts display the model's SHAP analysis results, including SHAP visualizations of feature contributions.
        """)

        # 计算 SHAP 值
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)

            # 生成 SHAP force plot
            st.header("1. SHAP Force Plot")
            html_output = shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                features_df.iloc[0, :],
                show=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{html_output.html()}</body>"
            st.components.v1.html(shap_html, height=400)

            # 生成 SHAP summary plot
            st.header("2. SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(4, 3))
            shap.summary_plot(shap_values, features_df, plot_type="dot", show=False)
            plt.title("SHAP Values for Each Feature")
            st.pyplot(fig)

            # 生成 SHAP feature importance plot
            st.header("3. SHAP Feature Importance")
            fig, ax = plt.subplots(figsize=(4, 3))
            shap.summary_plot(shap_values, features_df, plot_type="bar", show=False)
            plt.title("SHAP Values for Each Feature")
            st.pyplot(fig)

            # 生成 SHAP decision plot
            st.header("4. SHAP Decision Plot")
            fig, ax = plt.subplots(figsize=(4, 3))
            shap.decision_plot(explainer.expected_value, shap_values[0, :], features_df.iloc[0, :], show=False)
            plt.title("SHAP Decision Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during SHAP visualization: {e}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 预测精度和精密度图
if true_value > 0 and prediction is not None:
    st.header("📊 Prediction Accuracy and Precision")
    st.write("Display the model's absolute and relative accuracy, as well as precision within ±30% and ±3.")

    # 计算绝对和相对精度
    absolute_accuracy = abs(prediction - true_value)
    relative_accuracy = abs((prediction - true_value) / true_value) * 100 if true_value != 0 else 0

    # 计算 ±30% 和 ±3 的精密度
    precision_30_percent = abs(prediction - true_value) <= (0.3 * true_value)
    precision_3 = abs(prediction - true_value) <= 3

    # 显示精度和精密度指标
    st.subheader("Accuracy and Precision Metrics")
    st.write(f"Absolute Accuracy: {absolute_accuracy:.2f} mg/L")
    st.write(f"Relative Accuracy: {relative_accuracy:.2f}%")
    st.write(f"Precision within ±30%: {'Yes' if precision_30_percent else 'No'}")
    st.write(f"Precision within ±3 mg/L: {'Yes' if precision_3 else 'No'}")

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_value, prediction, alpha=0.5, color='blue', label='Prediction')
    ax.plot([0, max(true_value, prediction)], [0, max(true_value, prediction)], color='red', linestyle='--', label='Ideal Line')
    ax.set_xlabel('True Values (mg/L)')
    ax.set_ylabel('Predicted Values (mg/L)')
    ax.set_title('Prediction Accuracy and Precision')
    ax.legend()

    # 添加指标信息
    textstr = '\n'.join((
        f'Absolute Accuracy: {absolute_accuracy:.2f} mg/L',
        f'Relative Accuracy: {relative_accuracy:.2f}%',
        f'Precision within ±30%: {"Yes" if precision_30_percent else "No"}',
        f'Precision within ±3 mg/L: {"Yes" if precision_3 else "No"}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    # 显示图表
    st.pyplot(fig)

# 页脚
st.markdown("---")
st.header("Summary")
st.write("""
Through this page, you can:
1. Perform real-time predictions using input feature values.
2. Gain an intuitive understanding of the model's SHAP analysis results, including SHAP visualizations of feature contributions.
3. If a true value is provided, the model's absolute and relative accuracy, as well as precision within ±30% and ±3, will also be displayed.
These analyses help to deeply understand the model's prediction logic and the importance of features.
""")
