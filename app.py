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

model, _ = load_model()

def adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat):
    """ปรับ probability ให้ 3 ปัจจัยหลักมีผลมากขึ้น"""

    # 1. ระยะเวลาใช้งาน — ยิ่งนาน ยิ่งลด risk
    if tenure_days >= 730:
        tenure_factor = -0.30
    elif tenure_days >= 365:
        tenure_factor = -0.15
    elif tenure_days >= 100:
        tenure_factor = -0.05
    else:
        tenure_factor = +0.15   # ใหม่มาก = เสี่ยง

    # 2. ระดับแพ็กเกจ — สูงกว่า ลด risk
    tier_factor = {
        "enterprise": -0.25,
        "premium":    -0.15,
        "standard":   -0.05,
        "basic":      +0.10,
    }.get(plan_tier, 0)

    # 3. ความพึงพอใจ — มีผลมากที่สุด
    if avg_sat >= 4.5:
        sat_factor = -0.35
    elif avg_sat >= 4.0:
        sat_factor = -0.20
    elif avg_sat >= 3.0:
        sat_factor = -0.05
    elif avg_sat >= 2.0:
        sat_factor = +0.15
    else:
        sat_factor = +0.30   # พอใจน้อยมาก = เสี่ยงสูง

    adjusted = raw_prob + tenure_factor + tier_factor + sat_factor
    return float(np.clip(adjusted, 0.02, 0.98))

# ── Header ─────────────────────────────────────────────────────────
st.title("📊 ทำนายความเสี่ยง Churn")
st.markdown("กรอกข้อมูลลูกค้า แล้วกด **ทำนาย** เพื่อดูความเสี่ยงการยกเลิกบริการ")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 ข้อมูลลูกค้า")
    tenure_days       = st.number_input("ระยะเวลาใช้งาน (วัน)",
                            min_value=0, max_value=5000, value=730, step=30,
                            help="มากกว่า 100 วัน = ลดความเสี่ยง | มากกว่า 2 ปี = ลดมาก")
    plan_tier         = st.selectbox("ระดับแพ็กเกจ",
                            ["basic", "standard", "premium", "enterprise"],
                            index=3,
                            help="แพ็กเกจสูง = ลงทุนมาก = ความเสี่ยงต่ำกว่า")

    has_churn_history = st.selectbox("เคยยกเลิกบริการมาก่อน?",
                            [0, 1], format_func=lambda x: "เคย ⚠️" if x else "ไม่เคย ✅")
    n_churn_events    = st.number_input("จำนวนครั้งที่ยกเลิกบริการ",
                            min_value=0, max_value=20, value=0, step=1)
    industry          = st.selectbox("อุตสาหกรรม",
                            ["Cybersecurity", "DevTools", "EdTech", "FinTech", "HealthTech"])
    country           = st.selectbox("ประเทศ",
                            ["AU", "CA", "DE", "FR", "IN", "UK", "US"])

with col2:
    st.subheader("⭐ ความพึงพอใจ & Support")
    avg_sat           = st.slider("ความพึงพอใจเฉลี่ย (1–5)",
                            min_value=1.0, max_value=5.0, value=4.5, step=0.1,
                            help="ปัจจัยสำคัญที่สุด — 4.5+ ลดความเสี่ยงได้มาก")
    escalation_rate   = st.slider("สัดส่วน Ticket ที่บานปลาย",
                            min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.0f%%")
    usage_trend       = st.slider("แนวโน้มการใช้งาน",
                            min_value=-1.0, max_value=1.0, value=0.3, step=0.05,
                            help="บวก = ใช้งานมากขึ้น ✅ | ลบ = ใช้งานลดลง ⚠️")

st.divider()
predict_btn = st.button("🔍 ทำนายความเสี่ยง", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "tenure_days":         tenure_days,
        "avg_sat":             avg_sat,
        "avg_resolution":      12.0,
        "escalation_rate":     escalation_rate,
        "annual_ratio":        1.2,
        "n_urgent":            1,
        "beta_ratio":          0.05,
        "sub_churn_rate":      sub_churn_rate,
        "error_rate":          0.01,
        "n_trials":            1,
        "n_subs":              5,
        "n_escalated":         max(0, int(escalation_rate * 10)),
        "n_tickets":           10,
        "has_churn_history":   has_churn_history,
        "n_churn_events":      n_churn_events,
        "avg_seats":           10.0,
        "avg_first_resp":      3.0,
        "sat_std":             0.5,
        "usage_trend":         usage_trend,
        "usage_days":          max(1, int(tenure_days * 0.7)),
        "pct_no_sat":          0.1,
        "ticket_acceleration": 0.0,
        "recent_escalations":  0,
        "min_sat":             max(1.0, avg_sat - 0.8),
        "referral_source":     "organic",
        "plan_tier":           plan_tier,
        "industry":            industry,
        "country":             country,
    }])

    try:
        raw_prob   = model.predict_proba(input_data)[0][1]
        churn_prob = adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat)

        st.divider()

        if churn_prob < 0.30:
            st.success("## ✅ ความเสี่ยงต่ำ")
            st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยังคงใช้บริการต่อไป**")
        elif churn_prob < 0.55:
            st.warning("## 🟡 ความเสี่ยงปานกลาง")
            st.markdown("ควร **ติดตามและดูแล** ลูกค้ารายนี้เป็นพิเศษ")
        else:
            st.error("## ⚠️ ความเสี่ยงสูง")
            st.markdown("ลูกค้ารายนี้มีแนวโน้มที่จะ **ยกเลิกบริการ** — ควรดำเนินการเชิงรุกทันที")

        st.metric("โอกาสยกเลิกบริการ", f"{churn_prob:.1%}")
        st.progress(float(churn_prob))

        # ── ปัจจัยที่มีผล ──
        st.markdown("**📋 ปัจจัยที่มีผลต่อการทำนาย**")
        f1, f2, f3 = st.columns(3)
        with f1:
            icon = "✅" if tenure_days >= 100 else "⚠️"
            st.metric(f"{icon} ระยะเวลาใช้งาน", f"{tenure_days} วัน")
        with f2:
            tier_icon = {"enterprise":"✅","premium":"✅","standard":"🟡","basic":"⚠️"}.get(plan_tier,"")
            st.metric(f"{tier_icon} แพ็กเกจ", plan_tier.capitalize())
        with f3:
            sat_icon = "✅" if avg_sat >= 4.0 else ("🟡" if avg_sat >= 3.0 else "⚠️")
            st.metric(f"{sat_icon} ความพึงพอใจ", f"{avg_sat:.1f} / 5.0")

        if churn_prob < 0.30:
            st.info("💡 **คำแนะนำ:** รักษาคุณภาพ service และโปรแกรม loyalty ต่อไป")
        elif churn_prob < 0.55:
            st.warning("💡 **คำแนะนำ:** ติดต่อ Customer Success เพื่อ check-in และเสนอ upgrade")
        else:
            st.error("💡 **คำแนะนำ:** โทรหาลูกค้าทันที เสนอส่วนลดหรือ retention package")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.exception(e)

st.divider()
st.caption("ขับเคลื่อนด้วย Gradient Boosting · สร้างด้วย Streamlit")
