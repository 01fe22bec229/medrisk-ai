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
    import json as _json
    with open(model_path, 'r') as _f:
        _check = _json.load(_f)
    if "learner" not in _check and "prediction" in _check:
        raise ValueError("best_model.json is a placeholder — please export a real XGBoost model.")

    model          = XGBClassifier()
    model.load_model(model_path)
    scaler         = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    feature_names  = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    MODEL_LOADED   = True
    print("✅ XGBoost model loaded successfully — ML predictions active.")
    print(f"   Model expects {len(feature_names)} features: {feature_names}")
except Exception as _e:
    MODEL_LOADED = False
    print(f"⚠️  Model load failed: {_e}")
    print("    Running in rule-based fallback mode.")

# ── MAPPING: index.html field names → CSV/model feature names ────────────────
# Your index.html sends keys like 'A1_Year_of_Study'.
# Your CSV (and therefore feature_names.pkl) may use different names.
# Edit the RIGHT side of each line below to match your CSV column names exactly.
FIELD_MAP = {
    'A1_Year_of_Study'               : 'A1_Year_of_Study',
    'A2_Age_Group'                   : 'A2_Age_Group',
    'A3_Gender'                      : 'A3_Gender',
    'A4_Residence'                   : 'A4_Residence',
    'A5_Academic_Performance'        : 'A5_Academic_Performance',
    'A6_Healthcare_Family'           : 'A6_Healthcare_Family',
    'B7_SelfMed_Past6Months'        : 'B7_SelfMed_Past6Months',
    'B7a_Frequency'                  : 'B7a_Frequency',
    'B8_Duration_Without_Doctor'     : 'B8_Duration_Without_Doctor',
    'B9_Formal_Drug_Education'       : 'B9_Formal_Drug_Education',
    'B10_Frequency_SelfMed'          : 'B10_Frequency_SelfMed',
    'B11_Reasons'                    : 'B11_Reasons',
    'B12_Med_Types'                  : 'B12_Med_Types',
    'B13_Antibiotic_Course_Completion': 'B13_Antibiotic_Course_Completion',
    'B14_Medication_Sources'         : 'B14_Medication_Sources',
    'B15_Checked_Info_Source'        : 'B15_Checked_Info_Source',
    'B15a_Info_Sources_Specified'    : 'B15a_Info_Sources_Specified',
    'B16_Source_Check_Frequency'     : 'B16_Source_Check_Frequency',
    'C15_PSS4_Q1_Unexpected'        : 'C15_PSS4_Q1_Unexpected',
    'C15_PSS4_Q2_Control'           : 'C15_PSS4_Q2_Control',
    'C15_PSS4_Q3_Nervous'           : 'C15_PSS4_Q3_Nervous',
    'C15_PSS4_Q4_Difficulties'      : 'C15_PSS4_Q4_Difficulties',
    'C15_PSS4_Total'                 : 'C15_PSS4_Total',
    'C15_PSS4_Category'              : 'C15_PSS4_Category',
    'C16_Stress_Influences_SelfMed'  : 'C16_Stress_Influences_SelfMed',
    'C17_SelfMed_For_Stress_Anxiety' : 'C17_SelfMed_For_Stress_Anxiety',
    'C17a_How_Often'                 : 'C17a_How_Often',
    'C18_SelfMed_For_Sleep'          : 'C18_SelfMed_For_Sleep',
    'C19_Peer_Influence'             : 'C19_Peer_Influence',
    'C20_Social_Media_Med_Decisions' : 'C20_Social_Media_Med_Decisions',
    'D21_Attitude_Minor_Ailments'    : 'D21_Attitude_Minor_Ailments',
    'D21_Attitude_Confidence'        : 'D21_Attitude_Confidence',
    'D21_Attitude_Safe_Overall'      : 'D21_Attitude_Safe_Overall',
    'D21_Attitude_Saves_Time_Justified': 'D21_Attitude_Saves_Time_Justified',
    'D21_Attitude_Training_Helps'    : 'D21_Attitude_Training_Helps',
    'D22_Experienced_Relief'         : 'D22_Experienced_Relief',
    'D23_Aware_of_Risks'             : 'D23_Aware_of_Risks',
    'D24_Risks_Recognised'           : 'D24_Risks_Recognised',
    'D25_Risks_Personally_Experienced': 'D25_Risks_Personally_Experienced',
    'D26_Experienced_ADR'            : 'D26_Experienced_ADR',
    'D26a_ADR_Reported'              : 'D26a_ADR_Reported',
    'D27_Aware_Pharmacovigilance'    : 'D27_Aware_Pharmacovigilance',
    'E30_Uses_Medical_Apps'          : 'E30_Uses_Medical_Apps',
    'E31_Digital_Tool_Frequency'     : 'E31_Digital_Tool_Frequency',
    'E32_Searches_Symptoms_Online'   : 'E32_Searches_Symptoms_Online',
    'E33_Trusted_Online_Sources'     : 'E33_Trusted_Online_Sources',
    'E34_Online_Purchase_No_Rx'      : 'E34_Online_Purchase_No_Rx',
}

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

# ── DEBUG ROUTE — visit /debug-full to see exactly what is loaded ─────────────
@app.route('/debug-full')
def debug_full():
    import traceback
    result = {}

    try:
        from xgboost import XGBClassifier as _X
        _m = _X()
        _m.load_model(os.path.join(BASE_DIR, "best_model.json"))
        result["1_model"] = "OK"
    except Exception as e:
        result["1_model"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        _s = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        result["2_scaler"] = "OK"
    except Exception as e:
        result["2_scaler"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        _enc = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
        result["3_encoders"] = f"OK — keys: {list(_enc.keys())}"
    except Exception as e:
        result["3_encoders"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        _fn = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
        result["4_feature_names"] = f"OK — {len(_fn)} features"
        result["4_feature_names_list"] = _fn   # ← shows every exact column name
    except Exception as e:
        result["4_feature_names"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        _dummy = np.zeros((1, len(_fn)))
        _scaled = _s.transform(_dummy)
        _pred = _m.predict(_scaled)
        _proba = _m.predict_proba(_scaled)
        result["5_dummy_prediction"] = f"OK — class={int(_pred[0])}, proba={_proba[0].tolist()}"
    except Exception as e:
        result["5_dummy_prediction"] = f"FAILED: {traceback.format_exc()}"

    result["MODEL_LOADED_flag"] = MODEL_LOADED
    return jsonify(result)


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
    if freq == 'Daily':    score += 18
    elif freq == 'Weekly':  score += 12
    elif freq == 'Monthly': score += 6

    dur = str(form_data.get('B8_Duration_Without_Doctor') or '')
    if dur == '>5 days':    score += 14
    elif dur == '3-5 days': score += 8

    atb = str(form_data.get('B13_Antibiotic_Course_Completion') or '')
    if 'No' in atb or 'early' in atb.lower():                                score += 12
    if str(form_data.get('D26_Experienced_ADR') or '')               == 'Yes': score += 10
    if str(form_data.get('C20_Social_Media_Med_Decisions') or '')    == 'Yes': score += 8
    if str(form_data.get('E34_Online_Purchase_No_Rx') or '')         == 'Yes': score += 10
    if str(form_data.get('C19_Peer_Influence') or '')              == 'Yes used': score += 6
    if str(form_data.get('C18_SelfMed_For_Sleep') or '')             == 'Yes': score += 6
    if str(form_data.get('B16_Source_Check_Frequency') or '')      == 'Never': score += 8
    elif str(form_data.get('B16_Source_Check_Frequency') or '')    == 'Rarely': score += 4

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
            try:
                # Step 1 — Remap form field names → model feature names using FIELD_MAP
                remapped = {}
                for form_key, model_key in FIELD_MAP.items():
                    remapped[model_key] = str(form_data.get(form_key, 'Unknown'))

                # Step 2 — Build a numeric row directly (avoids pandas dtype conflicts)
                # Each feature is encoded to an integer immediately — no string DataFrame
                numeric_row = []
                for col in feature_names:
                    val = remapped.get(col, 'Unknown')
                    if col in label_encoders:
                        le = label_encoders[col]
                        if val in le.classes_:
                            numeric_row.append(int(le.transform([val])[0]))
                        elif 'Unknown' in le.classes_:
                            numeric_row.append(int(le.transform(['Unknown'])[0]))
                        else:
                            numeric_row.append(0)
                    else:
                        # Numeric column — convert directly
                        try:
                            numeric_row.append(float(val))
                        except (ValueError, TypeError):
                            numeric_row.append(0.0)

                # Step 3 — Scale and predict using the clean numeric array
                X_input  = np.array([numeric_row], dtype=float)
                X_scaled = scaler.transform(X_input)
                # Step 4 — Predict
                pred_encoded  = int(model.predict(X_scaled)[0])
                probabilities = model.predict_proba(X_scaled)[0]

                # Step 5 — Map encoded prediction back to label string
                target_col = 'Risk_Label'
                if target_col in label_encoders:
                    le         = label_encoders[target_col]
                    pred_label = le.inverse_transform([pred_encoded])[0]
                    probs_dict = {'low': 0.0, 'moderate': 0.0, 'high': 0.0}
                    for i, cls in enumerate(le.classes_):
                        key = cls.lower().replace(' risk', '').strip()
                        if key in probs_dict:
                            probs_dict[key] = float(probabilities[i])
                else:
                    pred_label = str(pred_encoded)
                    probs_dict = {'low': 0.0, 'moderate': 0.0, 'high': 0.0}

                # Step 6 — Derive a 0–100 risk score from probabilities
                risk_score = int(round(
                    probs_dict.get('high', 0)     * 85 +
                    probs_dict.get('moderate', 0) * 45 +
                    probs_dict.get('low', 0)      * 10
                ))
                risk_score = max(5, min(100, risk_score))
                method     = 'XGBoost Model'

            except Exception as ml_err:
                # XGBoost path failed — log error and fall back
                import traceback
                print(f"❌ XGBoost prediction failed: {ml_err}")
                traceback.print_exc()
                risk_score, pred_label, probs_dict = rule_based_predict(form_data)
                method = f'Rule-Based Fallback (XGBoost error: {ml_err})'

        else:
            # Model never loaded — use rule-based
            risk_score, pred_label, probs_dict = rule_based_predict(form_data)
            method = 'Rule-Based Fallback (model not loaded)'

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


# ── MODEL STATUS (quick check) ────────────────────────────────────────────────
@app.route('/model-status')
def model_status():
    if MODEL_LOADED:
        return jsonify({
            "status": "XGBoost loaded OK",
            "features": len(feature_names),
            "feature_names": feature_names
        })
    return jsonify({"status": "Model NOT loaded — running rule-based fallback"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
