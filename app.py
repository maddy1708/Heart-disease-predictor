"""
Heart Attack Risk Prediction - Dual-Mode Application
---------------------------------------------------
1. Serverless / WSGI / ASGI API Mode:
   Exports top-level `app`, `application`, and `handler` callables for Vercel,
   AWS Lambda, Gunicorn, uWSGI, and REST API consumers.

2. Interactive Streamlit Web UI Mode:
   Interactive clinical dashboard launched via `streamlit run app.py` (or directly `python app.py`).
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

# Setup project root and module paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.preprocessing import clean_data
except ImportError:
    from preprocessing import clean_data

MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Global In-Memory Model Cache for API mode
_MODEL_CACHE: Dict[str, Any] = {}
_PREPROCESSOR_CACHE: Optional[Any] = None


def get_available_models() -> List[str]:
    """Return sorted list of all trained classification model names."""
    if not MODELS_DIR.exists():
        return []
    files = list(MODELS_DIR.glob("*.pkl"))
    model_names = [f.stem for f in files if f.name not in ["preprocessor.pkl", "metadata.pkl"]]
    return sorted(model_names)


def load_preprocessor():
    """Load and cache the fitted ColumnTransformer preprocessor."""
    global _PREPROCESSOR_CACHE
    if _PREPROCESSOR_CACHE is None and PREPROCESSOR_PATH.exists():
        _PREPROCESSOR_CACHE = joblib.load(PREPROCESSOR_PATH)
    return _PREPROCESSOR_CACHE


def load_model(model_name: str = "voting_ensemble"):
    """Load and cache a trained model by name with fallback to default."""
    global _MODEL_CACHE
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        available = get_available_models()
        if not available:
            return None
        model_name = "voting_ensemble" if "voting_ensemble" in available else available[0]
        model_path = MODELS_DIR / f"{model_name}.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
        _MODEL_CACHE[model_name] = model
        return model

    return None


def parse_patient_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize, parse, and validate patient input data.
    Supports both raw 'Blood Pressure' string (e.g. '140/90') and separated 'Systolic BP'/'Diastolic BP'.
    """
    parsed = {}

    # 1. Parse Blood Pressure
    if "Blood Pressure" in data and isinstance(data["Blood Pressure"], str):
        match = re.search(r"(\d+)\s*/\s*(\d+)", data["Blood Pressure"])
        if match:
            parsed["Systolic BP"] = float(match.group(1))
            parsed["Diastolic BP"] = float(match.group(2))
        else:
            parsed["Systolic BP"] = float(data.get("Systolic BP", 130))
            parsed["Diastolic BP"] = float(data.get("Diastolic BP", 82))
    else:
        parsed["Systolic BP"] = float(data.get("Systolic BP", 130))
        parsed["Diastolic BP"] = float(data.get("Diastolic BP", 82))

    # 2. Demographics & Continuous Metrics
    parsed["Age"] = float(data.get("Age", 52))
    sex_val = str(data.get("Sex", "Male")).strip().capitalize()
    parsed["Sex"] = sex_val if sex_val in ["Male", "Female"] else "Male"
    parsed["Cholesterol"] = float(data.get("Cholesterol", 215))
    parsed["Heart Rate"] = float(data.get("Heart Rate", 72))
    parsed["Income"] = int(float(data.get("Income", 120000)))
    parsed["BMI"] = float(data.get("BMI", 26.2))
    parsed["Triglycerides"] = float(data.get("Triglycerides", 175))
    parsed["Exercise Hours Per Week"] = float(data.get("Exercise Hours Per Week", 4.0))
    parsed["Sedentary Hours Per Day"] = float(data.get("Sedentary Hours Per Day", 6.5))
    parsed["Physical Activity Days Per Week"] = int(float(data.get("Physical Activity Days Per Week", 3)))
    parsed["Sleep Hours Per Day"] = int(float(data.get("Sleep Hours Per Day", 7)))
    parsed["Stress Level"] = int(float(data.get("Stress Level", 5)))

    # 3. Categorical & Binary Clinical Flags
    def to_binary(val: Any) -> int:
        if isinstance(val, bool):
            return 1 if val else 0
        if str(val).strip().lower() in ["1", "true", "yes", "smoker", "positive", "y"]:
            return 1
        return 0

    parsed["Diabetes"] = to_binary(data.get("Diabetes", 0))
    parsed["Family History"] = to_binary(data.get("Family History", 0))
    parsed["Smoking"] = to_binary(data.get("Smoking", 0))
    parsed["Alcohol Consumption"] = to_binary(data.get("Alcohol Consumption", 0))
    parsed["Previous Heart Problems"] = to_binary(data.get("Previous Heart Problems", 0))
    parsed["Medication Use"] = to_binary(data.get("Medication Use", 0))

    # Obesity flag: 1 if BMI >= 30 or explicitly set
    parsed["Obesity"] = 1 if (parsed["BMI"] >= 30.0 or to_binary(data.get("Obesity", 0))) else 0

    diet_val = str(data.get("Diet", "Average")).strip().capitalize()
    parsed["Diet"] = diet_val if diet_val in ["Healthy", "Average", "Unhealthy"] else "Average"

    return parsed


def generate_clinical_recommendations(data: Dict[str, Any]) -> List[str]:
    """Generate personalized clinical & lifestyle recommendations based on patient vitals."""
    recs = []
    sbp = data["Systolic BP"]
    dbp = data["Diastolic BP"]
    chol = data["Cholesterol"]
    trig = data["Triglycerides"]
    bmi = data["BMI"]
    smoking = data["Smoking"]
    exercise = data["Exercise Hours Per Week"]
    sedentary = data["Sedentary Hours Per Day"]
    stress = data["Stress Level"]
    diabetes = data["Diabetes"]
    prev_heart = data["Previous Heart Problems"]

    if sbp >= 140 or dbp >= 90:
        recs.append(f"Stage 2 Hypertension ({int(sbp)}/{int(dbp)} mmHg): Consult a physician for clinical anti-hypertensive evaluation and initiate sodium restriction.")
    elif sbp >= 130 or dbp >= 85:
        recs.append(f"Elevated Blood Pressure ({int(sbp)}/{int(dbp)} mmHg): Monitor BP bi-weekly and adopt the DASH dietary pattern.")

    if chol >= 240:
        recs.append(f"High Total Cholesterol ({int(chol)} mg/dL): Lipid panel assessment and dietary soluble fiber enhancement recommended.")
    elif chol >= 200:
        recs.append(f"Borderline Cholesterol ({int(chol)} mg/dL): Limit saturated fats and trans fats.")

    if trig >= 200:
        recs.append(f"Elevated Triglycerides ({int(trig)} mg/dL): Reduce refined sugars, simple carbohydrates, and alcohol consumption.")

    if bmi >= 30.0:
        recs.append(f"Obesity Range BMI ({bmi:.1f}): Guided weight reduction strategy through caloric balance is strongly advised.")
    elif bmi >= 25.0:
        recs.append(f"Overweight BMI ({bmi:.1f}): Aim for 5-7% gradual body weight reduction.")

    if smoking == 1:
        recs.append("Active Tobacco Use: Cessation substantially halts coronary atherosclerotic plaque progression.")

    if exercise < 2.5:
        recs.append("Insufficient Aerobic Activity: Target at least 150 minutes of moderate-intensity cardio weekly.")

    if sedentary >= 8.0:
        recs.append("Prolonged Sedentary Time: Take 3-5 minute active walking intervals every waking hour.")

    if stress >= 7:
        recs.append("Elevated Perceived Stress: Implement restorative sleep hygiene and structured stress mitigation practices.")

    if diabetes == 1:
        recs.append("Diabetes Management: Maintain strict glycemic control to preserve coronary micro-vasculature.")

    if prev_heart == 1:
        recs.append("Prior Cardiac Event: Adhere rigorously to prescribed cardioprotective regimens and cardiologist follow-ups.")

    if not recs:
        recs.append("Optimal cardiovascular baseline! Maintain your current balanced dietary and exercise habits.")

    return recs


def predict_cardiovascular_risk(raw_patient_data: Dict[str, Any], model_name: str = "voting_ensemble") -> Dict[str, Any]:
    """Execute prediction pipeline on single patient data dictionary."""
    preprocessor = load_preprocessor()
    model = load_model(model_name)

    if preprocessor is None or model is None:
        return {
            "status": "error",
            "error": "Model artifacts not found. Please train models first with 'python src/train.py'."
        }

    try:
        clean_patient = parse_patient_payload(raw_patient_data)
        df_input = pd.DataFrame([clean_patient])
        cleaned_df = clean_data(df_input)
        transformed = preprocessor.transform(cleaned_df)

        prediction = int(model.predict(transformed)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(transformed)[0][1])
        else:
            probability = 0.85 if prediction == 1 else 0.15

        if probability >= 0.50:
            risk_level = "Elevated Risk"
            risk_color = "#e63946"
        elif probability >= 0.30:
            risk_level = "Moderate Risk"
            risk_color = "#f4a261"
        else:
            risk_level = "Low Risk"
            risk_color = "#2a9d8f"

        recommendations = generate_clinical_recommendations(clean_patient)

        return {
            "status": "success",
            "prediction": prediction,
            "probability": round(probability, 4),
            "probability_percent": f"{probability * 100:.2f}%",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "model_used": model_name,
            "recommendations": recommendations,
            "parsed_vitals": {
                "Age": clean_patient["Age"],
                "Blood_Pressure": f"{int(clean_patient['Systolic BP'])}/{int(clean_patient['Diastolic BP'])} mmHg",
                "Cholesterol": f"{int(clean_patient['Cholesterol'])} mg/dL",
                "BMI": clean_patient["BMI"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Inference execution failed: {str(e)}"
        }


# ==============================================================================
# 🌐 WSGI & Serverless API Handler (Vercel, AWS Lambda, Gunicorn, uWSGI)
# ==============================================================================

class UnifiedServerlessApp:
    """
    High-compatibility WSGI / ASGI / Serverless Application callable.
    Exports 'app', 'application', and 'handler' variables.
    """

    CORS_HEADERS = [
        ("Content-Type", "application/json"),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization")
    ]

    def __call__(self, *args, **kwargs):
        # 1. AWS Lambda / Serverless Event: (event, context)
        if len(args) == 2 and isinstance(args[0], dict) and "httpMethod" in args[0]:
            return self.handle_lambda_event(args[0], args[1])

        # 2. WSGI standard: (environ, start_response)
        if len(args) >= 2 and callable(args[1]):
            return self.handle_wsgi(args[0], args[1])

        # 3. ASGI standard: async (scope, receive, send)
        if len(args) == 3 and isinstance(args[0], dict) and "type" in args[0]:
            return self.handle_asgi(args[0], args[1], args[2])

        return {"status": "ok", "service": "CardioGuard ML API"}

    def handle_wsgi(self, environ, start_response):
        path = environ.get("PATH_INFO", "/").rstrip("/") or "/"
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if method == "OPTIONS":
            start_response("200 OK", self.CORS_HEADERS)
            return [b""]

        if path in ["/predict", "/api/predict"] and method == "POST":
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
                body = environ["wsgi.input"].read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                model_name = data.get("model", "voting_ensemble")
                result = predict_cardiovascular_risk(data, model_name=model_name)
                status_code = "200 OK" if result.get("status") == "success" else "400 Bad Request"
                start_response(status_code, self.CORS_HEADERS)
                return [json.dumps(result, indent=2).encode("utf-8")]
            except Exception as e:
                start_response("400 Bad Request", self.CORS_HEADERS)
                return [json.dumps({"status": "error", "error": f"Invalid JSON payload: {str(e)}"}).encode("utf-8")]

        if path in ["/health", "/api/health"]:
            start_response("200 OK", self.CORS_HEADERS)
            return [json.dumps({"status": "healthy", "service": "CardioGuard ML API"}).encode("utf-8")]

        # Root Overview & API Documentation
        info = {
            "service": "🫀 CardioGuard Heart Attack Risk Prediction API",
            "status": "online",
            "version": "1.0.0",
            "available_models": get_available_models(),
            "endpoints": {
                "POST /predict": "Predict cardiovascular risk from patient JSON payload",
                "GET /health": "Health check status"
            },
            "sample_request": {
                "Age": 55, "Sex": "Male", "Cholesterol": 240, "Blood Pressure": "145/92",
                "Heart Rate": 80, "Diabetes": 1, "Smoking": 1, "BMI": 31.5,
                "Triglycerides": 280, "model": "voting_ensemble"
            }
        }
        start_response("200 OK", self.CORS_HEADERS)
        return [json.dumps(info, indent=2).encode("utf-8")]

    def handle_lambda_event(self, event, context):
        method = event.get("httpMethod", "GET").upper()
        path = event.get("path", "/").rstrip("/") or "/"
        body = event.get("body", "{}")

        cors = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}

        if path in ["/predict", "/api/predict"] and method == "POST":
            try:
                data = json.loads(body) if isinstance(body, str) else (body or {})
                model_name = data.get("model", "voting_ensemble")
                result = predict_cardiovascular_risk(data, model_name=model_name)
                return {"statusCode": 200, "headers": cors, "body": json.dumps(result)}
            except Exception as e:
                return {"statusCode": 400, "headers": cors, "body": json.dumps({"status": "error", "error": str(e)})}

        return {
            "statusCode": 200,
            "headers": cors,
            "body": json.dumps({"status": "healthy", "service": "CardioGuard API"})
        }

    async def handle_asgi(self, scope, receive, send):
        if scope["type"] == "http":
            body_bytes = b""
            more_body = True
            while more_body:
                msg = await receive()
                body_bytes += msg.get("body", b"")
                more_body = msg.get("more_body", False)

            path = scope.get("path", "/").rstrip("/") or "/"
            method = scope.get("method", "GET").upper()

            if path in ["/predict", "/api/predict"] and method == "POST":
                try:
                    data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                    result = predict_cardiovascular_risk(data, data.get("model", "voting_ensemble"))
                except Exception as e:
                    result = {"status": "error", "error": str(e)}
            else:
                result = {"status": "healthy", "service": "CardioGuard API"}

            res_content = json.dumps(result).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"], [b"access-control-allow-origin", b"*"]]
            })
            await send({"type": "http.response.body", "body": res_content})


# Standard exports for Vercel, AWS Lambda, Gunicorn, uWSGI
app = UnifiedServerlessApp()
application = app
handler = app


# ==============================================================================
# 🎨 Interactive Streamlit Web UI Mode
# ==============================================================================

def render_streamlit_ui():
    """Render the full interactive Streamlit web dashboard."""
    import streamlit as st

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
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🫀 CardioGuard: Heart Attack Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Driven Cardiovascular Risk Assessment & Clinical Prevention Insights</div>', unsafe_allow_html=True)

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
        st.markdown("### 📊 About this System")
        st.markdown("""
        Trained on cardiovascular profiles using:
        - **Hemodynamic Feature Engineering**: Pulse Pressure & Mean Arterial Pressure (MAP)
        - **Metabolic Composites**: Lipid Indices & Atherogenic Proxies
        - **Soft Voting Ensemble**: Calibrated multi-model probability aggregation
        - **5-Fold Cross-Validation**: Validated generalization
        """)

        st.divider()
        st.info("ℹ️ **Disclaimer**: This tool is an academic decision-support prototype and does not replace professional medical diagnosis.")

    preprocessor = load_preprocessor()
    model = load_model(selected_model_name)

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

    st.markdown("### 📋 Enter Patient Clinical Information")

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
                                    help="Normal: <120, Elevated: 120-129, Stage 1: 130-139, Stage 2: 140+")
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=60, max_value=120, value=82, step=1,
                                     help="Normal: <80, Stage 1: 80-89, Stage 2: 90+")
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
        patient_payload = {
            "Age": age, "Sex": sex, "Cholesterol": cholesterol, "Heart Rate": heart_rate,
            "Diabetes": diabetes, "Family History": family_history, "Smoking": smoking,
            "Obesity": obesity, "Alcohol Consumption": alcohol,
            "Exercise Hours Per Week": exercise_hours, "Diet": diet,
            "Previous Heart Problems": previous_heart_problems,
            "Medication Use": medication_use, "Stress Level": stress_level,
            "Sedentary Hours Per Day": sedentary_hours, "Income": income,
            "BMI": bmi, "Triglycerides": triglycerides,
            "Physical Activity Days Per Week": physical_activity_days,
            "Sleep Hours Per Day": sleep_hours,
            "Systolic BP": systolic_bp, "Diastolic BP": diastolic_bp
        }

        result = predict_cardiovascular_risk(patient_payload, model_name=selected_model_name)

        if result.get("status") == "success":
            probability = result["probability"]
            risk_level = result["risk_level"]

            st.markdown("## 📊 Assessment Results")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if probability >= 0.50:
                    st.error(f"### ⚠️ {risk_level}")
                    st.markdown("**Status**: High probability of cardiovascular event.")
                elif probability >= 0.30:
                    st.warning(f"### ⚡ {risk_level}")
                    st.markdown("**Status**: Moderate cardiovascular risk factors detected.")
                else:
                    st.success(f"### ✅ {risk_level}")
                    st.markdown("**Status**: Favorable cardiovascular profile.")

                st.metric("Predicted Risk Probability", result["probability_percent"])
                st.progress(min(1.0, max(0.0, probability)))

            with res_col2:
                st.markdown("#### 🔍 Flagged Risk Factors & Clinical Recommendations")
                for rec in result["recommendations"]:
                    st.markdown(f"• {rec}")
        else:
            st.error(f"Error executing prediction pipeline: {result.get('error')}")


def is_running_streamlit() -> bool:
    """Return True only if actively executing within a Streamlit server runtime."""
    try:
        from streamlit.runtime import exists as runtime_exists
        return runtime_exists()
    except Exception:
        return False


if __name__ == "__main__":
    if is_running_streamlit():
        render_streamlit_ui()
    else:
        # If user runs 'python app.py' directly, launch Streamlit automatically
        import subprocess
        print("=" * 60)
        print("Launching CardioGuard Streamlit Web Dashboard...")
        print("=" * 60)
        subprocess.run(["streamlit", "run", str(Path(__file__).resolve())])
elif is_running_streamlit():
    render_streamlit_ui()