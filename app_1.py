import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Patient Readmission Risk Scorer",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .risk-HIGH { background-color: #FCEBEB; color: #A32D2D; padding: 4px 12px;
                 border-radius: 6px; font-weight: 600; font-size: 13px; }
    .risk-MEDIUM { background-color: #FAEEDA; color: #854F0B; padding: 4px 12px;
                   border-radius: 6px; font-weight: 600; font-size: 13px; }
    .risk-LOW { background-color: #EAF3DE; color: #3B6D11; padding: 4px 12px;
                border-radius: 6px; font-weight: 600; font-size: 13px; }
    .rec-box { background-color: #f8f9fa; border-left: 4px solid #1D9E75;
               padding: 12px 16px; border-radius: 4px; margin-top: 12px; }
    .section-header { color: #666; font-size: 13px; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.05em;
                      margin-top: 1.2rem; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists("model.joblib"):
        return joblib.load("model.joblib")
    return None

model_data = load_model()
FEATURES = model_data["features"] if model_data else []
rf = model_data["model"] if model_data else None

def score_row(row):
    if rf is not None:
        X = pd.DataFrame([row])[FEATURES].fillna(0)
        return float(rf.predict_proba(X)[0][1])
    score = -2.8
    score += row.get('number_inpatient', 0) * 0.45
    score += row.get('number_emergency', 0) * 0.28
    score += (row.get('age_num', 65) / 100) * 0.30
    score += (row.get('time_in_hospital', 4) / 14) * 0.25
    score += (row.get('number_diagnoses', 6) / 16) * 0.20
    score += (row.get('num_medications', 12) / 30) * 0.18
    score += (row.get('num_lab_procedures', 40) / 100) * 0.15
    score += row.get('drug_changes', 0) * 0.10
    score += row.get('insulin_used', 0) * 0.12
    score += row.get('a1c_high', 0) * 0.08
    return 1 / (1 + np.exp(-score))

def get_risk(prob):
    if prob >= 0.30: return "HIGH"
    if prob >= 0.15: return "MEDIUM"
    return "LOW"

def get_rec(risk):
    return {
        "HIGH":   "Follow-up call within 48h + care coordinator referral",
        "MEDIUM": "Schedule 7-day follow-up appointment",
        "LOW":    "Standard discharge with routine follow-up"
    }[risk]

st.title("🏥 Patient Readmission Risk Scorer")
st.markdown("30-day readmission risk scoring tool for hospital case management teams.")
st.divider()

tab1, tab2 = st.tabs(["📋 Batch — Today's Discharges", "👤 Individual Patient"])

# ── TAB 1: BATCH ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("Upload a CSV of today's discharged patients to generate a ranked call list.")

    template = pd.DataFrame([
        {"patient_id":"P001","patient_name":"Jane Smith","age_num":72,"time_in_hospital":6,
         "num_lab_procedures":55,"num_procedures":2,"num_medications":18,"number_outpatient":0,
         "number_emergency":2,"number_inpatient":3,"number_diagnoses":9,"drug_changes":2,
         "admission_type_id":1,"discharge_disposition_id":1,"admission_source_id":7,
         "insulin_used":1,"diabetes_on_med":1,"med_changed":0,"a1c_tested":0,"a1c_high":0,"gender_male":0},
        {"patient_id":"P002","patient_name":"John Lee","age_num":55,"time_in_hospital":3,
         "num_lab_procedures":35,"num_procedures":1,"num_medications":10,"number_outpatient":1,
         "number_emergency":0,"number_inpatient":1,"number_diagnoses":5,"drug_changes":1,
         "admission_type_id":2,"discharge_disposition_id":1,"admission_source_id":7,
         "insulin_used":0,"diabetes_on_med":1,"med_changed":1,"a1c_tested":1,"a1c_high":0,"gender_male":1},
        {"patient_id":"P003","patient_name":"Maria Garcia","age_num":38,"time_in_hospital":1,
         "num_lab_procedures":20,"num_procedures":0,"num_medications":5,"number_outpatient":0,
         "number_emergency":0,"number_inpatient":0,"number_diagnoses":3,"drug_changes":0,
         "admission_type_id":3,"discharge_disposition_id":1,"admission_source_id":1,
         "insulin_used":0,"diabetes_on_med":0,"med_changed":0,"a1c_tested":1,"a1c_high":0,"gender_male":0},
    ])

    st.download_button("Download CSV template", template.to_csv(index=False),
                       "patient_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        for feat in FEATURES:
            if feat.startswith("race_") and feat not in df.columns:
                df[feat] = 0

        df["readmit_probability"] = df.apply(score_row, axis=1)
        df["risk_level"]          = df["readmit_probability"].apply(get_risk)
        df["recommendation"]      = df["risk_level"].apply(get_rec)
        df["expected_savings"]    = df["readmit_probability"].apply(lambda p: f"${p*0.25*15000:,.0f}")

        df_sorted = df.sort_values("readmit_probability", ascending=False).reset_index(drop=True)
        df_sorted.index += 1

        st.markdown("### Today's discharge summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients",  len(df_sorted))
        c2.metric("High Risk",       (df_sorted["risk_level"]=="HIGH").sum())
        c3.metric("Medium Risk",     (df_sorted["risk_level"]=="MEDIUM").sum())
        c4.metric("Total Expected Savings", f"${df_sorted['readmit_probability'].apply(lambda p: p*0.25*15000).sum():,.0f}")

        st.divider()
        st.markdown("### Prioritized call list")
        budget = st.slider("Daily call budget", 1, len(df_sorted), min(10, len(df_sorted)))
        flagged = df_sorted.head(budget).copy()
        flagged["readmit_probability"] = flagged["readmit_probability"].apply(lambda x: f"{x*100:.1f}%")

        display_cols = ["patient_name","readmit_probability","risk_level","recommendation","expected_savings"] \
                       if "patient_name" in df.columns else \
                       ["readmit_probability","risk_level","recommendation","expected_savings"]

        out = flagged[display_cols].copy()
        out.columns = [c.replace("_"," ").title() for c in out.columns]
        st.dataframe(out, use_container_width=True)

        st.download_button("Download full ranked list", df_sorted.to_csv(index=True),
                           "ranked_call_list.csv", "text/csv")
    else:
        st.info("Upload a CSV to generate today's ranked call list. Download the template above to see the required format.")

# ── TAB 2: INDIVIDUAL ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("Score an individual patient at the point of discharge.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Patient demographics</div>', unsafe_allow_html=True)
        age              = st.number_input("Age", 0, 100, 65, key="i_age")
        gender_male      = st.selectbox("Gender", ["Female","Male"], key="i_gen") == "Male"
        time_in_hospital = st.number_input("Days in Hospital", 1, 30, 4, key="i_time")

        st.markdown('<div class="section-header">Prior visit history</div>', unsafe_allow_html=True)
        number_inpatient  = st.number_input("Prior Inpatient Visits", 0, 20, 1, key="i_inp")
        number_emergency  = st.number_input("Prior ER Visits", 0, 20, 0, key="i_er")
        number_outpatient = st.number_input("Prior Outpatient Visits", 0, 20, 0, key="i_out")

    with c2:
        st.markdown('<div class="section-header">Clinical complexity</div>', unsafe_allow_html=True)
        number_diagnoses   = st.number_input("# Diagnoses", 1, 16, 6, key="i_diag")
        num_medications    = st.number_input("# Medications", 0, 80, 12, key="i_meds")
        num_lab_procedures = st.number_input("# Lab Procedures", 0, 130, 40, key="i_labs")
        num_procedures     = st.number_input("# Procedures", 0, 6, 1, key="i_proc")
        drug_changes       = st.number_input("# Drug Changes", 0, 10, 1, key="i_dc")

        st.markdown('<div class="section-header">Diabetes management</div>', unsafe_allow_html=True)
        insulin_used    = st.checkbox("Insulin prescribed", key="i_ins")
        diabetes_on_med = st.checkbox("On diabetes medication", value=True, key="i_dm")
        med_changed     = st.checkbox("Medication changed this visit", key="i_mc")
        a1c_tested      = st.checkbox("A1C test ordered", key="i_a1ct")
        a1c_high        = st.checkbox("A1C > 8%", key="i_a1ch")

    st.divider()

    if st.button("Calculate Risk", type="primary", use_container_width=True, key="i_btn"):
        race_feats = {c: 0 for c in FEATURES if c.startswith("race_")}
        patient = {
            "age_num": age, "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
            "num_medications": num_medications, "number_outpatient": number_outpatient,
            "number_emergency": number_emergency, "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses, "drug_changes": drug_changes,
            "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
            "insulin_used": int(insulin_used), "diabetes_on_med": int(diabetes_on_med),
            "med_changed": int(med_changed), "a1c_tested": int(a1c_tested),
            "a1c_high": int(a1c_high), "gender_male": int(gender_male), **race_feats
        }
        prob = score_row(patient)
        risk = get_risk(prob)
        rec  = get_rec(risk)

        c1, c2, c3 = st.columns(3)
        c1.metric("Readmission Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level", risk)
        c3.metric("Expected Savings if Called", f"${prob*0.25*15000:,.0f}")

        st.markdown(f'<span class="risk-{risk}">{risk} RISK</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box">📋 <strong>Recommendation:</strong> {rec}</div>', unsafe_allow_html=True)

        st.markdown("#### Key risk drivers")
        drivers = []
        if number_inpatient >= 3:   drivers.append(f"High prior inpatient visits ({number_inpatient})")
        if number_emergency >= 2:   drivers.append(f"Multiple prior ER visits ({number_emergency})")
        if number_diagnoses >= 8:   drivers.append(f"High comorbidity burden ({number_diagnoses} diagnoses)")
        if num_medications >= 15:   drivers.append(f"Polypharmacy risk ({num_medications} medications)")
        if time_in_hospital >= 7:   drivers.append(f"Extended hospital stay ({time_in_hospital} days)")
        if a1c_high:                drivers.append("Poor glycemic control (A1C > 8%)")
        if not drivers:             drivers.append("No major individual risk factors identified")
        for d in drivers:
            st.markdown(f"- {d}")
