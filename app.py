from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder=".")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOADED = False
model = None
scaler = None
label_encoders = None
feature_names = None

# ── LOAD ML ARTIFACTS SAFELY ─────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier

    model_path = os.path.join(BASE_DIR, "best_model.json")
    # Validate it's a real XGBoost model (not a placeholder JSON)
    import json as _json
    with open(model_path, 'r') as _f:
        _check = _json.load(_f)
    if "learner" not in _check and "prediction" in _check:
        raise ValueError("best_model.json is a placeholder — please export a real XGBoost model.")

    model = XGBClassifier()
    model.load_model(model_path)
    scaler         = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    feature_names  = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    MODEL_LOADED   = True
    print("✅ XGBoost model loaded successfully — ML predictions active.")
except Exception as _e:
    MODEL_LOADED = False
    print(f"⚠️  Model load failed: {_e}")
    print("    Running in rule-based fallback mode until a real model is supplied.")

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

# ── GENERATE INSIGHTS & RECOMMENDATIONS ───────────────────────────────────────
def generate_insights(form_data, label):
    insights = []
    pss = int(form_data.get('C15_PSS4_Total') or 0)
    if pss >= 11:
        insights.append(f"High psychological stress (PSS-4: {pss}/16) strongly drives self-medication decisions.")
    elif pss >= 6:
        insights.append(f"Moderate stress (PSS-4: {pss}/16) is influencing medication self-management.")

    freq = str(form_data.get('B10_Frequency_SelfMed') or '')
    if freq in ('Daily', 'Weekly'):
        insights.append(f"{freq} self-medication indicates a habitual, unsupervised drug-use pattern.")

    dur = str(form_data.get('B8_Duration_Without_Doctor') or '')
    if dur == '>5 days':
        insights.append("Self-medicating >5 days without consultation raises misdiagnosis and dependency risk.")

    atb = str(form_data.get('B13_Antibiotic_Course_Completion') or '')
    if 'No' in atb or 'early' in atb.lower():
        insights.append("Incomplete antibiotic courses are a key driver of antimicrobial resistance.")

    if str(form_data.get('D26_Experienced_ADR') or '') == 'Yes':
        insights.append("Prior adverse drug reaction history significantly elevates future self-medication risk.")

    if str(form_data.get('C20_Social_Media_Med_Decisions') or '') == 'Yes':
        insights.append("Social-media-influenced medication decisions bypass clinical validation.")

    if str(form_data.get('E34_Online_Purchase_No_Rx') or '') == 'Yes':
        insights.append("Purchasing prescription medicines online without an Rx violates drug-control regulations.")

    if str(form_data.get('C17_SelfMed_For_Stress_Anxiety') or '') == 'Yes':
        insights.append("Self-medicating for anxiety/stress without professional guidance can mask serious conditions.")

    if str(form_data.get('C18_SelfMed_For_Sleep') or '') == 'Yes':
        insights.append("Sleep-related self-medication risks dependency and rebound insomnia without supervision.")

    if not insights:
        insights.append("Risk profile computed from complete survey response. Continue monitoring behaviour patterns.")

    recs_map = {
        'High Risk': [
            "Consult your pharmacology faculty immediately about current self-medication practices.",
            "Never purchase prescription medicines online without a valid doctor's prescription.",
            "Complete every antibiotic course fully — stopping early breeds drug-resistant bacteria.",
            "Report adverse drug reactions via PvPI (India's pharmacovigilance portal).",
            "Seek professional counselling for stress/sleep — avoid self-medicating for these conditions."
        ],
        'Moderate Risk': [
            "Review WHO guidelines on rational drug use and responsible self-medication.",
            "Always verify drug information via pharmacist advice or approved medical textbooks.",
            "Reduce self-medication frequency and consult a doctor for any recurring symptoms.",
            "Build awareness of drug interactions and how to recognise adverse reactions early.",
        ],
        'Low Risk': [
            "Maintain your habit of checking reliable sources before every self-medication episode.",
            "Continue completing antibiotic courses fully — never stop at symptom resolution.",
            "Share safe self-medication knowledge with peers to lower collective risk.",
            "Stay updated on pharmacovigilance initiatives and participate in ADR reporting."
        ]
    }
    recs = recs_map.get(label, recs_map['Moderate Risk'])
    return insights, recs


# ── RULE-BASED FALLBACK ────────────────────────────────────────────────────────
def rule_based_predict(form_data):
    score = 0
    pss = int(form_data.get('C15_PSS4_Total') or 0)
    if pss >= 11: score += 20
    elif pss >= 6: score += 10

    freq = str(form_data.get('B10_Frequency_SelfMed') or '')
    if freq == 'Daily':   score += 18
    elif freq == 'Weekly': score += 12
    elif freq == 'Monthly': score += 6

    dur = str(form_data.get('B8_Duration_Without_Doctor') or '')
    if dur == '>5 days':    score += 14
    elif dur == '3-5 days': score += 8

    atb = str(form_data.get('B13_Antibiotic_Course_Completion') or '')
    if 'No' in atb or 'early' in atb.lower(): score += 12
    if str(form_data.get('D26_Experienced_ADR') or '')               == 'Yes': score += 10
    if str(form_data.get('C20_Social_Media_Med_Decisions') or '')     == 'Yes': score += 8
    if str(form_data.get('E34_Online_Purchase_No_Rx') or '')          == 'Yes': score += 10
    if str(form_data.get('C19_Peer_Influence') or '')                 == 'Yes used': score += 6
    if str(form_data.get('C18_SelfMed_For_Sleep') or '')              == 'Yes': score += 6
    if str(form_data.get('B16_Source_Check_Frequency') or '')         == 'Never': score += 8
    elif str(form_data.get('B16_Source_Check_Frequency') or '')       == 'Rarely': score += 4

    score = min(score, 100)
    label = 'High Risk' if score >= 60 else 'Moderate Risk' if score >= 30 else 'Low Risk'
    if label == 'High Risk':
        probs = {'low': 0.05, 'moderate': 0.20, 'high': 0.75}
    elif label == 'Moderate Risk':
        probs = {'low': 0.20, 'moderate': 0.60, 'high': 0.20}
    else:
        probs = {'low': 0.70, 'moderate': 0.25, 'high': 0.05}
    return score, label, probs


# ── PREDICT ENDPOINT ──────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.json or {}

        if MODEL_LOADED:
            # ── XGBoost path ──────────────────────────────────────────────────
            raw_df = pd.DataFrame([form_data])
            processed_df = pd.DataFrame(columns=feature_names)

            for col in feature_names:
                processed_df[col] = raw_df[col].astype(str) if col in raw_df.columns else "Unknown"

            for col in processed_df.columns:
                if col in label_encoders:
                    le  = label_encoders[col]
                    val = str(processed_df.loc[0, col])
                    if val in le.classes_:
                        processed_df.loc[0, col] = le.transform([val])[0]
                    elif "Unknown" in le.classes_:
                        processed_df.loc[0, col] = le.transform(["Unknown"])[0]
                    else:
                        processed_df.loc[0, col] = 0
                else:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)

            X_scaled = scaler.transform(processed_df.astype(float))
            pred_encoded  = int(model.predict(X_scaled)[0])
            probabilities = model.predict_proba(X_scaled)[0]

            # Map probabilities to named dict using the label encoder class order
            target_col = "Risk_Label"
            if target_col in label_encoders:
                le = label_encoders[target_col]
                pred_label = le.inverse_transform([pred_encoded])[0]
                # Build {low, moderate, high} dict regardless of encoder order
                probs_dict = {'low': 0.0, 'moderate': 0.0, 'high': 0.0}
                for i, cls in enumerate(le.classes_):
                    key = cls.lower().replace(' risk', '').strip()
                    if key in probs_dict:
                        probs_dict[key] = float(probabilities[i])
            else:
                pred_label = str(pred_encoded)
                probs_dict = {'low': 0.0, 'moderate': 0.0, 'high': 0.0}

            # Derive a 0-100 risk score from the probability distribution
            risk_score = int(round(
                probs_dict.get('high', 0) * 85 +
                probs_dict.get('moderate', 0) * 45 +
                probs_dict.get('low', 0) * 10
            ))
            risk_score = max(5, min(100, risk_score))
            method = 'XGBoost Model'

        else:
            # ── Rule-based fallback ───────────────────────────────────────────
            risk_score, pred_label, probs_dict = rule_based_predict(form_data)
            method = 'Rule-Based Fallback'

        insights, recommendations = generate_insights(form_data, pred_label)

        return jsonify({
            'status':          'success',
            'risk_label':      pred_label,
            'risk_score':      risk_score,
            'probs':           probs_dict,
            'confidence':      max(probs_dict.values()),
            'insights':        insights,
            'recommendations': recommendations,
            'method':          method
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/model-status')
def model_status():
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model("best_model.json")
        return {"status": "XGBoost loaded OK"}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
