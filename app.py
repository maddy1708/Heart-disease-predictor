"""
Heart Attack Risk Prediction - Interactive Streamlit Web Application
Provides real-time clinical risk assessment, probability scores, risk factor analysis,
and actionable lifestyle recommendations using trained machine learning pipelines.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.preprocessing import clean_data
except ImportError:
    from preprocessing import clean_data

DATA_PATH = PROJECT_ROOT / "data" / "heart_attack_prediction.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"


@st.cache_resource
def load_model_and_preprocessor(model_name: str = "voting_ensemble"):
    """Load the preprocessor and selected classification model."""
    if not PREPROCESSOR_PATH.exists():
        return None, None

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        available = list(MODELS_DIR.glob("*.pkl"))
        available = [p for p in available if p.name not in ["preprocessor.pkl", "metadata.pkl"]]
        if not available:
            return None, None
        model_path = available[0]

    model = joblib.load(model_path)
    return model, preprocessor


def get_available_models():
    """List available trained models."""
    if not MODELS_DIR.exists():
        return []
    files = list(MODELS_DIR.glob("*.pkl"))
    model_names = [f.stem for f in files if f.name not in ["preprocessor.pkl", "metadata.pkl"]]
    return sorted(model_names)


def main():
    st.set_page_config(
        page_title="CardioGuard | Heart Attack Risk Prediction",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #e63946;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #457b9d;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #e63946;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🫀 CardioGuard: Heart Attack Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Driven Cardiovascular Risk Assessment & Prevention Insights</div>', unsafe_allow_html=True)

    available_models = get_available_models()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        if available_models:
            default_name = "voting_ensemble" if "voting_ensemble" in available_models else available_models[0]
            default_idx = available_models.index(default_name)
            selected_model_name = st.selectbox(
                "Select Classifier",
                available_models,
                index=default_idx,
                format_func=lambda x: x.replace("_", " ").title()
            )
        else:
            selected_model_name = "voting_ensemble"
            st.warning("⚠️ No trained models found in `models/`.")

        st.divider()
        st.markdown("### 📊 About this Model")
        st.markdown("""
        Trained on clinical profiles using:
        - **Standardized Preprocessing**: Imputation & Scaling
        - **Feature Engineering**: Hemodynamic & Metabolic Indices
        - **Ensemble Validation**: 5-Fold Stratified Cross-Validation
        """)

        st.divider()
        st.info("ℹ️ **Disclaimer**: This tool is an academic decision-support prototype and does not replace professional medical diagnosis.")

    model, preprocessor = load_model_and_preprocessor(selected_model_name)

    if model is None or preprocessor is None:
        st.error("🚨 Trained model artifacts not found! Please train models first by executing:")
        st.code("python src/train.py", language="bash")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models..."):
                try:
                    from src.train import main as run_train
                    run_train()
                    st.success("Models trained successfully! Please reload the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training: {e}")
        return

    st.markdown("### 📋 Enter Patient Information")

    tab1, tab2, tab3 = st.tabs(["👤 Demographics & General", "🩺 Clinical & Vitals", "🏃 Lifestyle & Habits"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age (years)", min_value=18, max_value=95, value=52, step=1)
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col3:
            income = st.number_input("Annual Income ($ USD)", min_value=10000, max_value=300000, value=120000, step=5000)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=90, max_value=200, value=130, step=1,
                                    help="Normal: <120, Elevated: 120-129, High: 130+")
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=60, max_value=120, value=82, step=1,
                                     help="Normal: <80, High: 80+")
            heart_rate = st.slider("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=72, step=1)

        with col2:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=120, max_value=400, value=215, step=1,
                                          help="Desirable: <200, Borderline: 200-239, High: 240+")
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=800, value=175, step=1,
                                            help="Normal: <150, Borderline: 150-199, High: 200+")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=45.0, value=26.2, step=0.1,
                                  help="Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+")

        with col3:
            diabetes = st.selectbox("Diabetes Diagnosis", ["No", "Yes"])
            previous_heart_problems = st.selectbox("Previous Heart Incidents", ["No", "Yes"])
            medication_use = st.selectbox("Current Cardiac Medication", ["No", "Yes"])
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
            alcohol = st.selectbox("Regular Alcohol Consumption", ["No", "Yes"])
            diet = st.selectbox("Diet Quality", ["Healthy", "Average", "Unhealthy"])

        with col2:
            exercise_hours = st.slider("Exercise (Hours / Week)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
            physical_activity_days = st.slider("Active Days / Week", min_value=0, max_value=7, value=3, step=1)
            obesity = st.selectbox("Obesity Status", ["No", "Yes"] if bmi < 30 else ["Yes", "No"])

        with col3:
            stress_level = st.slider("Perceived Stress Level (1-10)", min_value=1, max_value=10, value=5, step=1)
            sedentary_hours = st.slider("Sedentary Hours / Day", min_value=0.0, max_value=16.0, value=6.5, step=0.5)
            sleep_hours = st.slider("Sleep (Hours / Day)", min_value=4, max_value=12, value=7, step=1)

    st.divider()

    # Prediction Action
    if st.button("🔍 Assess Cardiovascular Risk", type="primary", use_container_width=True):
        input_data = {
            "Age": age,
            "Sex": sex,
            "Cholesterol": cholesterol,
            "Heart Rate": heart_rate,
            "Diabetes": 1 if diabetes == "Yes" else 0,
            "Family History": 1 if family_history == "Yes" else 0,
            "Smoking": 1 if smoking == "Smoker" else 0,
            "Obesity": 1 if (obesity == "Yes" or bmi >= 30) else 0,
            "Alcohol Consumption": 1 if alcohol == "Yes" else 0,
            "Exercise Hours Per Week": float(exercise_hours),
            "Diet": diet,
            "Previous Heart Problems": 1 if previous_heart_problems == "Yes" else 0,
            "Medication Use": 1 if medication_use == "Yes" else 0,
            "Stress Level": int(stress_level),
            "Sedentary Hours Per Day": float(sedentary_hours),
            "Income": int(income),
            "BMI": float(bmi),
            "Triglycerides": int(triglycerides),
            "Physical Activity Days Per Week": int(physical_activity_days),
            "Sleep Hours Per Day": int(sleep_hours),
            "Systolic BP": float(systolic_bp),
            "Diastolic BP": float(diastolic_bp)
        }

        input_df = pd.DataFrame([input_data])
        cleaned_input_df = clean_data(input_df)

        try:
            transformed = preprocessor.transform(cleaned_input_df)
            prediction = int(model.predict(transformed)[0])

            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(transformed)[0][1])
            else:
                probability = 0.85 if prediction == 1 else 0.15

            # Display Results
            st.markdown("## 📊 Assessment Results")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if probability >= 0.50:
                    st.error("### ⚠️ Elevated Risk")
                    st.markdown("**Status**: High probability of cardiovascular event.")
                elif probability >= 0.30:
                    st.warning("### ⚡ Moderate Risk")
                    st.markdown("**Status**: Moderate cardiovascular risk factors detected.")
                else:
                    st.success("### ✅ Low Risk")
                    st.markdown("**Status**: Favorable cardiovascular profile.")

                st.metric("Predicted Risk Probability", f"{probability * 100:.1f}%")
                st.progress(min(1.0, max(0.0, probability)))

            with res_col2:
                st.markdown("#### 🔍 Flagged Risk Factors & Clinical Recommendations")
                risk_factors = []

                if systolic_bp >= 130 or diastolic_bp >= 85:
                    risk_factors.append(f"• **Elevated Blood Pressure** ({systolic_bp}/{diastolic_bp} mmHg): Monitor regularly and discuss DASH diet / sodium reduction with your physician.")
                if cholesterol >= 200:
                    risk_factors.append(f"• **Elevated Cholesterol** ({cholesterol} mg/dL): Consider lipid panel check and increase dietary soluble fiber.")
                if triglycerides >= 150:
                    risk_factors.append(f"• **High Triglycerides** ({triglycerides} mg/dL): Limit refined carbohydrates and sugary beverages.")
                if bmi >= 25.0:
                    risk_factors.append(f"• **Elevated BMI** ({bmi:.1f}): Aim for gradual weight management via balanced calorie deficit.")
                if smoking == "Smoker":
                    risk_factors.append("• **Tobacco Use**: Smoking dramatically increases arterial plaque formation. Seek smoking cessation support.")
                if exercise_hours < 2.5:
                    risk_factors.append("• **Low Aerobic Exercise**: Aim for at least 150 minutes of moderate physical activity per week.")
                if sedentary_hours >= 8.0:
                    risk_factors.append("• **High Sedentary Time**: Take brief 3-5 minute walking breaks every hour.")
                if stress_level >= 7:
                    risk_factors.append("• **High Stress Level**: Practice stress-reduction techniques (mindfulness, adequate rest).")
                if diabetes == "Yes":
                    risk_factors.append("• **Diabetes Diagnosis**: Maintain tight glycemic control to protect vascular integrity.")
                if previous_heart_problems == "Yes":
                    risk_factors.append("• **Prior Cardiac History**: Follow up routinely with a cardiologist.")

                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(rf)
                else:
                    st.success("• No critical lifestyle risk flags identified! Maintain current healthy dietary and exercise habits.")

        except Exception as e:
            st.error(f"Error executing prediction pipeline: {e}")


if __name__ == "__main__":
    main()