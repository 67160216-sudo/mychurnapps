import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ระบบทำนายการยกเลิกบริการ",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    import joblib, sys, types
    from sklearn.pipeline import Pipeline as _Pipeline
    dummy = types.ModuleType("Pipeline")
    dummy.Pipeline = _Pipeline
    sys.modules.setdefault("Pipeline", dummy)
    try:
        obj = joblib.load("gb_churn_model_fast.pkl")
    except Exception:
        with open("gb_churn_model_fast.pkl", "rb") as f:
            obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj.get("model", obj)
    return obj

model = load_model()

st.title("📊 ระบบทำนายการยกเลิกบริการ (Churn Prediction)")
st.markdown("กรอกข้อมูลลูกค้าด้านล่าง แล้วกด **ทำนาย** เพื่อดูความเสี่ยงการยกเลิกบริการ")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏢 ข้อมูลบัญชี")
    tenure_days       = st.number_input("ระยะเวลาใช้งาน (วัน)",         min_value=0,    max_value=5000,  value=365,  step=1)
    n_subs            = st.number_input("จำนวน Subscription",            min_value=0,    max_value=500,   value=5,    step=1)
    avg_seats         = st.number_input("จำนวนที่นั่งเฉลี่ย",            min_value=0.0,  max_value=500.0, value=10.0, step=0.5)
    has_churn_history = st.selectbox("เคยยกเลิกบริการมาก่อน?",          [0, 1], format_func=lambda x: "เคย" if x else "ไม่เคย")
    n_churn_events    = st.number_input("จำนวนครั้งที่ยกเลิกบริการ",     min_value=0,    max_value=100,   value=0,    step=1)
    plan_tier         = st.selectbox("ระดับแพ็กเกจ",                    ["basic", "standard", "premium", "enterprise"])
    referral_source   = st.selectbox("แหล่งที่มาของลูกค้า",              ["organic", "paid", "referral", "partner", "unknown"])
    industry          = st.selectbox("อุตสาหกรรม",                      ["Cybersecurity", "DevTools", "EdTech", "FinTech", "HealthTech"])
    country           = st.selectbox("ประเทศ",                          ["AU", "CA", "DE", "FR", "IN", "UK", "US"])

with col2:
    st.subheader("🎫 ข้อมูล Ticket")
    n_tickets           = st.number_input("จำนวน Ticket ทั้งหมด",         min_value=0,    max_value=5000,  value=20,  step=1)
    n_urgent            = st.number_input("Ticket ด่วน",                  min_value=0,    max_value=1000,  value=3,   step=1)
    n_escalated         = st.number_input("Ticket ที่ถูก Escalate",        min_value=0,    max_value=1000,  value=2,   step=1)
    escalation_rate     = st.number_input("อัตรา Escalation",             min_value=0.0,  max_value=1.0,   value=0.10,step=0.01, format="%.2f")
    ticket_acceleration = st.number_input("อัตราการเพิ่มของ Ticket",      min_value=-10.0,max_value=10.0,  value=0.0, step=0.1,  format="%.2f")
    recent_escalations  = st.number_input("Escalation ล่าสุด",            min_value=0,    max_value=100,   value=1,   step=1)
    n_trials            = st.number_input("จำนวนการทดลองใช้",             min_value=0,    max_value=100,   value=1,   step=1)

with col3:
    st.subheader("⭐ ความพึงพอใจและการใช้งาน")
    avg_sat        = st.slider("ความพึงพอใจเฉลี่ย (1–5)",          min_value=1.0, max_value=5.0,   value=3.5, step=0.1)
    min_sat        = st.slider("ความพึงพอใจต่ำสุด (1–5)",          min_value=1.0, max_value=5.0,   value=2.0, step=0.1)
    sat_std        = st.number_input("ค่าเบี่ยงเบนความพึงพอใจ",    min_value=0.0, max_value=3.0,   value=0.8, step=0.1,  format="%.2f")
    pct_no_sat     = st.number_input("% Ticket ที่ไม่มีการให้คะแนน",min_value=0.0, max_value=1.0,   value=0.2, step=0.01, format="%.2f")
    avg_resolution = st.number_input("เวลาแก้ปัญหาเฉลี่ย (ชม.)",   min_value=0.0, max_value=720.0, value=24.0,step=1.0)
    avg_first_resp = st.number_input("เวลาตอบกลับครั้งแรก (ชม.)",  min_value=0.0, max_value=720.0, value=4.0, step=0.5)
    usage_days     = st.number_input("วันที่มีการใช้งาน",           min_value=0,   max_value=5000,  value=300, step=1)
    usage_trend    = st.number_input("แนวโน้มการใช้งาน",            min_value=-10.0,max_value=10.0, value=0.0, step=0.1,  format="%.2f")
    annual_ratio   = st.number_input("อัตราส่วนรายปี",              min_value=0.0, max_value=10.0,  value=1.0, step=0.1,  format="%.2f")
    beta_ratio     = st.number_input("Beta Ratio",                  min_value=0.0, max_value=1.0,   value=0.1, step=0.01, format="%.2f")
    sub_churn_rate = st.number_input("อัตรา Churn ของ Subscription", min_value=0.0, max_value=1.0,  value=0.05,step=0.01, format="%.2f")
    error_rate     = st.number_input("อัตราข้อผิดพลาด",             min_value=0.0, max_value=1.0,   value=0.01,step=0.01, format="%.2f")

st.divider()
predict_btn = st.button("🔍 ทำนายการยกเลิกบริการ", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "tenure_days":         tenure_days,
        "avg_sat":             avg_sat,
        "avg_resolution":      avg_resolution,
        "escalation_rate":     escalation_rate,
        "annual_ratio":        annual_ratio,
        "n_urgent":            n_urgent,
        "beta_ratio":          beta_ratio,
        "sub_churn_rate":      sub_churn_rate,
        "error_rate":          error_rate,
        "n_trials":            n_trials,
        "n_subs":              n_subs,
        "n_escalated":         n_escalated,
        "n_tickets":           n_tickets,
        "has_churn_history":   has_churn_history,
        "n_churn_events":      n_churn_events,
        "avg_seats":           avg_seats,
        "avg_first_resp":      avg_first_resp,
        "sat_std":             sat_std,
        "usage_trend":         usage_trend,
        "usage_days":          usage_days,
        "pct_no_sat":          pct_no_sat,
        "ticket_acceleration": ticket_acceleration,
        "recent_escalations":  recent_escalations,
        "min_sat":             min_sat,
        "referral_source":     referral_source,
        "plan_tier":           plan_tier,
        "industry":            industry,
        "country":             country,
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        churn_prob = probability[1] if len(probability) > 1 else probability[0]

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ ความเสี่ยงสูง!")
                st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยกเลิกบริการ**")
            else:
                st.success("### ✅ ความเสี่ยงต่ำ")
                st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยังคงใช้บริการต่อไป**")

        with res_col2:
            st.metric(
                label="โอกาสยกเลิกบริการ",
                value=f"{churn_prob:.1%}",
                delta=f"{'ความเสี่ยงสูง' if churn_prob > 0.5 else 'ความเสี่ยงต่ำ'}",
                delta_color="inverse",
            )

        st.markdown("**ระดับความเสี่ยง**")
        st.progress(float(churn_prob))

        if churn_prob < 0.3:
            st.info("🟢 **ความเสี่ยงต่ำ** (< 30%) — ลูกค้ามีความพึงพอใจดี ควรรักษาความสัมพันธ์ต่อไป")
        elif churn_prob < 0.6:
            st.warning("🟡 **ความเสี่ยงปานกลาง** (30–60%) — ควรติดตามและหาวิธีเพิ่มความพึงพอใจ")
        else:
            st.error("🔴 **ความเสี่ยงสูง** (> 60%) — ต้องดำเนินการเชิงรุกทันที เพื่อป้องกันการยกเลิกบริการ")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.exception(e)

st.divider()
st.caption("ขับเคลื่อนด้วย Gradient Boosting · สร้างด้วย Streamlit")
