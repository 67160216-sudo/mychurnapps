import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ระบบทำนายการยกเลิกบริการ",
    page_icon="📊",
    layout="centered",
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
        return obj.get("model"), obj.get("threshold", 0.20)
    return obj, 0.20

model, THRESHOLD = load_model()

# ── Header ──────────────────────────────────────────────────────────
st.title("📊 ทำนายความเสี่ยง Churn")
st.markdown("กรอกข้อมูลลูกค้า แล้วกด **ทำนาย** เพื่อดูความเสี่ยงการยกเลิกบริการ")
st.divider()

# ── ฟีเจอร์สำคัญ 3 กลุ่ม (ลดจาก 28 → 10) ──────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 ข้อมูลลูกค้า")
    tenure_days       = st.number_input("ระยะเวลาใช้งาน (วัน)",        min_value=0,   max_value=5000, value=730,  step=30,
                                         help="ยิ่งนานยิ่งภักดี — ลูกค้าที่ใช้งาน 2+ ปีมีความเสี่ยงต่ำกว่า")
    sub_churn_rate    = st.slider("อัตรา Subscription ที่ยกเลิก",      min_value=0.0, max_value=1.0,  value=0.05, step=0.01,
                                         format="%.0f%%",
                                         help="ตัวบ่งชี้สำคัญที่สุด — ยิ่งต่ำยิ่งดี")
    has_churn_history = st.selectbox("เคยยกเลิกบริการมาก่อน?",         [0, 1], format_func=lambda x: "เคย ⚠️" if x else "ไม่เคย ✅")
    plan_tier         = st.selectbox("ระดับแพ็กเกจ",                   ["basic", "standard", "premium", "enterprise"])
    industry          = st.selectbox("อุตสาหกรรม",                     ["Cybersecurity", "DevTools", "EdTech", "FinTech", "HealthTech"])
    country           = st.selectbox("ประเทศ",                         ["AU", "CA", "DE", "FR", "IN", "UK", "US"])

with col2:
    st.subheader("⭐ ความพึงพอใจ & Support")
    avg_sat           = st.slider("ความพึงพอใจเฉลี่ย (1–5)",           min_value=1.0, max_value=5.0,  value=4.0,  step=0.1,
                                         help="คะแนนเฉลี่ยจาก Support Ticket")
    escalation_rate   = st.slider("สัดส่วน Ticket ที่บานปลาย",         min_value=0.0, max_value=1.0,  value=0.05, step=0.01,
                                         format="%.0f%%",
                                         help="ยิ่งต่ำยิ่งดี — ลูกค้าที่ไม่มี escalation เลยมีความเสี่ยงต่ำ")
    usage_trend       = st.slider("แนวโน้มการใช้งาน",                  min_value=-1.0,max_value=1.0,  value=0.2,  step=0.05,
                                         help="บวก = ใช้งานมากขึ้น | ลบ = ใช้งานลดลง (เสี่ยง)")
    n_churn_events    = st.number_input("จำนวนครั้งที่ยกเลิกบริการ",    min_value=0,   max_value=20,   value=0,    step=1)

# ── ค่า default สำหรับ features ที่ซ่อนอยู่ ────────────────────────
DEFAULTS = dict(
    avg_resolution=12.0, annual_ratio=1.2, n_urgent=1,
    beta_ratio=0.05, error_rate=0.01, n_trials=1,
    n_subs=5, n_escalated=int(escalation_rate * 10),
    n_tickets=10, avg_seats=10.0, avg_first_resp=3.0,
    sat_std=0.5, usage_days=int(tenure_days * 0.7),
    pct_no_sat=0.1, ticket_acceleration=0.0,
    recent_escalations=0, min_sat=max(1.0, avg_sat - 0.8),
    referral_source="organic",
)

st.divider()
predict_btn = st.button("🔍 ทำนายความเสี่ยง", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "tenure_days":         tenure_days,
        "avg_sat":             avg_sat,
        "avg_resolution":      DEFAULTS["avg_resolution"],
        "escalation_rate":     escalation_rate,
        "annual_ratio":        DEFAULTS["annual_ratio"],
        "n_urgent":            DEFAULTS["n_urgent"],
        "beta_ratio":          DEFAULTS["beta_ratio"],
        "sub_churn_rate":      sub_churn_rate,
        "error_rate":          DEFAULTS["error_rate"],
        "n_trials":            DEFAULTS["n_trials"],
        "n_subs":              DEFAULTS["n_subs"],
        "n_escalated":         DEFAULTS["n_escalated"],
        "n_tickets":           DEFAULTS["n_tickets"],
        "has_churn_history":   has_churn_history,
        "n_churn_events":      n_churn_events,
        "avg_seats":           DEFAULTS["avg_seats"],
        "avg_first_resp":      DEFAULTS["avg_first_resp"],
        "sat_std":             DEFAULTS["sat_std"],
        "usage_trend":         usage_trend,
        "usage_days":          DEFAULTS["usage_days"],
        "pct_no_sat":          DEFAULTS["pct_no_sat"],
        "ticket_acceleration": DEFAULTS["ticket_acceleration"],
        "recent_escalations":  DEFAULTS["recent_escalations"],
        "min_sat":             DEFAULTS["min_sat"],
        "referral_source":     DEFAULTS["referral_source"],
        "plan_tier":           plan_tier,
        "industry":            industry,
        "country":             country,
    }])

    try:
        churn_prob = model.predict_proba(input_data)[0][1]
        prediction = int(churn_prob >= THRESHOLD)

        st.divider()

        # ── ผลลัพธ์หลัก ──
        if churn_prob < 0.25:
            st.success("## ✅ ความเสี่ยงต่ำ")
            st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยังคงใช้บริการต่อไป**")
        elif churn_prob < 0.45:
            st.warning("## 🟡 ความเสี่ยงปานกลาง")
            st.markdown("ควร **ติดตามและดูแล** ลูกค้ารายนี้เป็นพิเศษ")
        else:
            st.error("## ⚠️ ความเสี่ยงสูง")
            st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยกเลิกบริการ** — ควรดำเนินการเชิงรุกทันที")

        # ── Metric & Progress ──
        st.metric("โอกาสยกเลิกบริการ", f"{churn_prob:.1%}")
        st.progress(float(churn_prob))

        # ── คำแนะนำ ──
        if churn_prob < 0.25:
            st.info("💡 **คำแนะนำ:** รักษาคุณภาพ service และโปรแกรม loyalty ต่อไป")
        elif churn_prob < 0.45:
            st.warning("💡 **คำแนะนำ:** ติดต่อ Customer Success เพื่อ check-in และเสนอ upgrade")
        else:
            st.error("💡 **คำแนะนำ:** โทรหาลูกค้าทันที เสนอส่วนลดหรือ retention package")

        st.caption(f"probability = {churn_prob:.4f} | threshold = {THRESHOLD:.2f}")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.exception(e)

st.divider()
st.caption("ขับเคลื่อนด้วย Gradient Boosting · สร้างด้วย Streamlit")
