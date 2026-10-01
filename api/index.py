"""
Vercel Serverless Function & API Handler for CardioGuard
Provides:
1. Interactive Web Dashboard on GET / (Accessible in any browser on Vercel)
2. JSON Prediction Endpoint on POST /predict & POST /api/predict
3. Health Check on GET /health & GET /api/health
"""

import os
import sys
import json
import re
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs

# Configure paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import joblib
import numpy as np
import pandas as pd

try:
    from src.preprocessing import clean_data
except ImportError:
    from preprocessing import clean_data

MODELS_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

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
    """Load and cache the preprocessor."""
    global _PREPROCESSOR_CACHE
    if _PREPROCESSOR_CACHE is None and PREPROCESSOR_PATH.exists():
        _PREPROCESSOR_CACHE = joblib.load(PREPROCESSOR_PATH)
    return _PREPROCESSOR_CACHE


def load_model(model_name: str = "voting_ensemble"):
    """Load and cache model."""
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
    """Sanitize and normalize input payload."""
    parsed = {}

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
    parsed["Obesity"] = 1 if (parsed["BMI"] >= 30.0 or to_binary(data.get("Obesity", 0))) else 0

    diet_val = str(data.get("Diet", "Average")).strip().capitalize()
    parsed["Diet"] = diet_val if diet_val in ["Healthy", "Average", "Unhealthy"] else "Average"

    return parsed


def generate_clinical_recommendations(data: Dict[str, Any]) -> List[str]:
    """Generate lifestyle recommendations based on patient vitals."""
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
        recs.append(f"Stage 2 Hypertension ({int(sbp)}/{int(dbp)} mmHg): Clinical anti-hypertensive evaluation and dietary sodium restriction recommended.")
    elif sbp >= 130 or dbp >= 85:
        recs.append(f"Elevated Blood Pressure ({int(sbp)}/{int(dbp)} mmHg): Monitor BP bi-weekly and adopt the DASH dietary pattern.")

    if chol >= 240:
        recs.append(f"High Total Cholesterol ({int(chol)} mg/dL): Lipid panel assessment and dietary soluble fiber enhancement advised.")
    elif chol >= 200:
        recs.append(f"Borderline Cholesterol ({int(chol)} mg/dL): Limit saturated fats and trans-fat intake.")

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
        recs.append("Diabetes Management: Maintain strict glycemic control (HbA1c < 7.0%) to preserve coronary vasculature.")

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
            "error": "Model artifacts not found. Please ensure models are trained and saved."
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
# 🌐 Vercel Serverless BaseHTTPRequestHandler
# ==============================================================================

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioGuard | Heart Attack Risk Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #e63946;
      --primary-dark: #b71c1c;
      --secondary: #1d3557;
      --accent: #457b9d;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 14px;
      --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px 16px;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      background: linear-gradient(135deg, #1d3557 0%, #0f172a 100%);
      color: white;
      padding: 32px;
      border-radius: var(--radius);
      margin-bottom: 24px;
      box-shadow: var(--shadow);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header-title h1 { font-size: 1.8rem; font-weight: 800; display: flex; align-items: center; gap: 10px; }
    .header-title p { color: #94a3b8; font-size: 0.95rem; margin-top: 4px; }
    .badge {
      background: rgba(230, 57, 70, 0.15);
      color: #ff6b6b;
      border: 1px solid rgba(230, 57, 70, 0.3);
      padding: 6px 14px;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.85rem;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
    }
    .card h2 {
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--secondary);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; color: #334155; }
    input, select {
      width: 100%;
      padding: 10px 14px;
      border: 1.5px solid var(--border);
      border-radius: 8px;
      font-size: 0.95rem;
      font-family: inherit;
      transition: all 0.2s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(69, 123, 157, 0.15);
    }
    .row { display: flex; gap: 12px; }
    .row > * { flex: 1; }
    .btn {
      width: 100%;
      background: var(--primary);
      color: white;
      border: none;
      padding: 14px;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 12px rgba(230, 57, 70, 0.25);
    }
    .btn:hover { background: var(--primary-dark); transform: translateY(-1px); }
    .result-box {
      display: none;
      padding: 24px;
      border-radius: var(--radius);
      margin-top: 16px;
      text-align: center;
    }
    .risk-score { font-size: 2.8rem; font-weight: 800; line-height: 1; margin: 12px 0; }
    .risk-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.9rem;
      color: white;
      margin-bottom: 12px;
    }
    .recs-list { text-align: left; margin-top: 20px; list-style-type: none; }
    .recs-list li {
      position: relative;
      padding: 8px 12px 8px 32px;
      background: #f8fafc;
      border-radius: 6px;
      margin-bottom: 8px;
      font-size: 0.88rem;
    }
    .recs-list li::before {
      content: "•";
      color: var(--primary);
      font-size: 1.4rem;
      position: absolute;
      left: 12px;
      top: 0px;
    }
    .code-box {
      background: #0f172a;
      color: #38bdf8;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 0.85rem;
      overflow-x: auto;
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>🫀 CardioGuard</h1>
        <p>High-Precision Machine Learning for Early Heart Attack Risk Stratification</p>
      </div>
      <div class="badge">98.97% Accuracy • 0.9994 ROC-AUC</div>
    </header>

    <div class="grid">
      <!-- Input Form -->
      <div class="card">
        <h2>📋 Enter Patient Parameters</h2>
        <form id="riskForm">
          <div class="form-group">
            <label>Classification Model</label>
            <select id="model">
              <option value="logistic_regression">Logistic Regression (98.97% Accuracy • Fast)</option>
              <option value="voting_ensemble" selected>Soft Voting Ensemble (97.09% Accuracy • Recommended)</option>
              <option value="gradient_boosting">Gradient Boosting (95.21% Accuracy)</option>
              <option value="xgboost">XGBoost / HistGB (94.92% Accuracy)</option>
              <option value="svm">Support Vector Machine (97.03% Accuracy)</option>
              <option value="random_forest">Random Forest (91.22% Accuracy)</option>
            </select>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Age (Years)</label>
              <input type="number" id="age" value="55" min="18" max="95" required>
            </div>
            <div class="form-group">
              <label>Sex</label>
              <select id="sex">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Systolic BP (mmHg)</label>
              <input type="number" id="systolic" value="145" min="80" max="220" required>
            </div>
            <div class="form-group">
              <label>Diastolic BP (mmHg)</label>
              <input type="number" id="diastolic" value="92" min="50" max="130" required>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Total Cholesterol (mg/dL)</label>
              <input type="number" id="cholesterol" value="240" min="100" max="450" required>
            </div>
            <div class="form-group">
              <label>Triglycerides (mg/dL)</label>
              <input type="number" id="triglycerides" value="220" min="40" max="700" required>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Body Mass Index (BMI)</label>
              <input type="number" id="bmi" value="31.2" step="0.1" min="15" max="50" required>
            </div>
            <div class="form-group">
              <label>Resting Heart Rate (bpm)</label>
              <input type="number" id="heart_rate" value="78" min="40" max="130" required>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Diabetes Diagnosis</label>
              <select id="diabetes">
                <option value="0">No</option>
                <option value="1" selected>Yes</option>
              </select>
            </div>
            <div class="form-group">
              <label>Previous Heart Incidents</label>
              <select id="prev_heart">
                <option value="0">No</option>
                <option value="1" selected>Yes</option>
              </select>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label>Smoking Status</label>
              <select id="smoking">
                <option value="0">Non-Smoker</option>
                <option value="1" selected>Active Smoker</option>
              </select>
            </div>
            <div class="form-group">
              <label>Exercise (Hours / Week)</label>
              <input type="number" id="exercise" value="2.0" step="0.5" min="0" max="25" required>
            </div>
          </div>

          <button type="submit" class="btn" id="submitBtn">🔍 Assess Cardiovascular Risk</button>
        </form>
      </div>

      <!-- Output Panel -->
      <div class="card">
        <h2>📊 Assessment Results</h2>
        <div id="loading" style="display:none; text-align:center; padding: 40px;">
          <p style="font-weight:600; color:var(--accent);">Executing machine learning pipeline...</p>
        </div>

        <div id="resultsPanel">
          <div style="text-align: center; color: var(--text-muted); padding: 40px 10px;">
            <p style="font-size: 3rem; margin-bottom: 12px;">🫀</p>
            <p style="font-weight: 600;">Fill in patient vitals and click "Assess Cardiovascular Risk" to compute probability score.</p>
          </div>
        </div>

        <div id="resultBox" class="result-box">
          <div id="riskBadge" class="risk-badge">Elevated Risk</div>
          <p style="color: var(--text-muted); font-size: 0.9rem; font-weight: 600;">PREDICTED RISK PROBABILITY</p>
          <div id="riskScore" class="risk-score">92.4%</div>
          <p id="modelInfo" style="font-size: 0.82rem; color: var(--text-muted);"></p>
          
          <h3 style="text-align:left; font-size: 0.95rem; margin-top: 24px; color: var(--secondary);">🔍 Clinical Findings & Lifestyle Recommendations:</h3>
          <ul id="recsList" class="recs-list"></ul>
        </div>

        <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);">
          <h3 style="font-size: 0.9rem; color: var(--secondary); margin-bottom: 6px;">🔌 Serverless REST API</h3>
          <p style="font-size: 0.8rem; color: var(--text-muted);">Send POST requests to <code>/api/predict</code> with patient JSON payloads:</p>
          <div class="code-box">curl -X POST https://heart-disease-predictor-t4zu.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 55, "Systolic BP": 145, "Diastolic BP": 92, "Cholesterol": 240, "model": "logistic_regression"}'</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    document.getElementById('riskForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const submitBtn = document.getElementById('submitBtn');
      const loading = document.getElementById('loading');
      const resultsPanel = document.getElementById('resultsPanel');
      const resultBox = document.getElementById('resultBox');
      
      submitBtn.disabled = true;
      loading.style.display = 'block';
      resultsPanel.style.display = 'none';
      resultBox.style.display = 'none';

      const payload = {
        model: document.getElementById('model').value,
        Age: parseFloat(document.getElementById('age').value),
        Sex: document.getElementById('sex').value,
        "Systolic BP": parseFloat(document.getElementById('systolic').value),
        "Diastolic BP": parseFloat(document.getElementById('diastolic').value),
        Cholesterol: parseFloat(document.getElementById('cholesterol').value),
        Triglycerides: parseFloat(document.getElementById('triglycerides').value),
        BMI: parseFloat(document.getElementById('bmi').value),
        "Heart Rate": parseFloat(document.getElementById('heart_rate').value),
        Diabetes: parseInt(document.getElementById('diabetes').value),
        "Previous Heart Problems": parseInt(document.getElementById('prev_heart').value),
        Smoking: parseInt(document.getElementById('smoking').value),
        "Exercise Hours Per Week": parseFloat(document.getElementById('exercise').value)
      };

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        loading.style.display = 'none';
        submitBtn.disabled = false;

        if (data.status === 'success') {
          resultBox.style.display = 'block';
          resultBox.style.backgroundColor = data.probability >= 0.5 ? '#fef2f2' : (data.probability >= 0.3 ? '#fffbeb' : '#f0fdf4');
          resultBox.style.border = `1.5px solid ${data.risk_color}`;

          const riskBadge = document.getElementById('riskBadge');
          riskBadge.textContent = data.risk_level;
          riskBadge.style.backgroundColor = data.risk_color;

          const riskScore = document.getElementById('riskScore');
          riskScore.textContent = data.probability_percent;
          riskScore.style.color = data.risk_color;

          document.getElementById('modelInfo').textContent = `Inference Model: ${data.model_used.replace(/_/g, ' ').toUpperCase()}`;

          const recsList = document.getElementById('recsList');
          recsList.innerHTML = '';
          data.recommendations.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            recsList.appendChild(li);
          });
        } else {
          resultsPanel.style.display = 'block';
          resultsPanel.innerHTML = `<div style="color: #e63946; padding: 20px; font-weight:600;">Error: ${data.error || 'Failed to compute prediction'}</div>`;
        }
      } catch (err) {
        loading.style.display = 'none';
        submitBtn.disabled = false;
        resultsPanel.style.display = 'block';
        resultsPanel.innerHTML = `<div style="color: #e63946; padding: 20px; font-weight:600;">Network Error: ${err.message}</div>`;
      }
    });
  </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    """Vercel Native Serverless Handler."""

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"

        if path in ["/health", "/api/health"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "CardioGuard ML API"}).encode("utf-8"))
            return

        if path in ["/models", "/api/models"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"available_models": get_available_models()}).encode("utf-8"))
            return

        # Serve Interactive Web Dashboard
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._send_cors()
        self.end_headers()
        self.wfile.write(HTML_DASHBOARD.encode("utf-8"))

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"

        if path in ["/predict", "/api/predict", "/"]:
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(body) if body else {}
                model_name = data.get("model", "voting_ensemble")
                result = predict_cardiovascular_risk(data, model_name=model_name)

                status_code = 200 if result.get("status") == "success" else 400
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": f"Invalid request: {str(e)}"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))


# Standard exports for compatibility
app = handler
application = handler
