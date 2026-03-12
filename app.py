import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ระบบวิเคราะห์ความเสี่ยง Churn",
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

def adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat, last_active, usage_trend):
    """ปรับ probability ให้สะท้อนความเป็นจริงในเชิงธุรกิจมากขึ้น"""
    
    # 1. ระยะเวลาใช้งาน
    tenure_factor = -0.30 if tenure_days >= 730 else (+0.15 if tenure_days < 100 else -0.05)
    
    # 2. ระดับแพ็กเกจ
    tier_factor = {"enterprise": -0.25, "premium": -0.15, "standard": -0.05, "basic": +0.10}.get(plan_tier, 0)
    
    # 3. ความพึงพอใจ
    sat_factor = -0.35 if avg_sat >= 4.5 else (+0.30 if avg_sat < 2.5 else -0.05)
    
    # 4. พฤติกรรมล่าสุด (ปัจจัยสำคัญที่เพิ่มเข้ามา)
    active_factor = +0.25 if last_active > 30 else -0.10 # หายไปนาน = เสี่ยงสูง
    trend_factor = -0.10 if usage_trend > 0.2 else (+0.15 if usage_trend < -0.2 else 0)

    adjusted = raw_prob + tenure_factor + tier_factor + sat_factor + active_factor + trend_factor
    
    # เก็บข้อมูลปัจจัยไว้แสดงกราฟ
    impact_data = {
        "Tenure": tenure_factor,
        "Plan Tier": tier_factor,
        "Satisfaction": sat_factor,
        "Inactivity": active_factor,
        "Usage Trend": trend_factor
    }
    
    return float(np.clip(adjusted, 0.01, 0.99)), impact_data

# ── Header ─────────────────────────────────────────────────────────
st.title("📊 Churn Risk Analytics")
st.markdown("วิเคราะห์ความเสี่ยงการยกเลิกบริการด้วย AI และ Business Intelligence")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 ข้อมูลพื้นฐาน")
    tenure_days = st.number_input("ระยะเวลาใช้งาน (วัน)", min_value=0, value=730, step=30)
    plan_tier = st.selectbox("ระดับแพ็กเกจ", ["basic", "standard", "premium", "enterprise"], index=3)
    monthly_spend = st.number_input("ยอดจ่ายรายเดือน ($)", min_value=0, value=150)
    industry = st.selectbox("อุตสาหกรรม", ["Cybersecurity", "DevTools", "EdTech", "FinTech", "HealthTech"])

with col2:
    st.subheader("⚡ พฤติกรรมล่าสุด")
    avg_sat = st.slider("ความพึงพอใจเฉลี่ย (1–5)", 1.0, 5.0, 4.5, step=0.1)
    last_active = st.slider("ไม่ได้ใช้งานมาแล้ว (วัน)", 0, 90, 5, help="สำคัญมาก: ยิ่งนานยิ่งเสี่ยง")
    usage_trend = st.slider("แนวโน้มการใช้งาน", -1.0, 1.0, 0.3, step=0.1, help="ค่าบวกคือใช้งานเพิ่มขึ้น")
    escalation_rate = st.slider("สัดส่วน Ticket ที่บานปลาย", 0.0, 1.0, 0.05, format="%.0f%%")

st.divider()
if st.button("🔍 ทำนายความเสี่ยง", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "tenure_days": tenure_days, "avg_sat": avg_sat, "avg_resolution": 12.0,
        "escalation_rate": escalation_rate, "annual_ratio": 1.2, "n_urgent": 1,
        "beta_ratio": 0.05, "sub_churn_rate": 0.05, "error_rate": 0.01,
        "n_trials": 1, "n_subs": 5, "n_escalated": int(escalation_rate * 10),
        "n_tickets": 10, "has_churn_history": 0, "n_churn_events": 0,
        "avg_seats": 10.0, "avg_first_resp": 3.0, "sat_std": 0.5,
        "usage_trend": usage_trend, "usage_days": max(1, int(tenure_days * 0.7)),
        "pct_no_sat": 0.1, "ticket_acceleration": 0.0, "recent_escalations": 0,
        "min_sat": max(1.0, avg_sat - 0.8), "referral_source": "organic",
        "plan_tier": plan_tier, "industry": industry, "country": "US",
    }])

    try:
        raw_prob = model.predict_proba(input_data)[0][1]
        churn_prob, impacts = adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat, last_active, usage_trend)

        # ── การแสดงผล ──
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            if churn_prob < 0.35:
                st.success(f"## ✅ ความเสี่ยงต่ำ ({churn_prob:.1%})")
            elif churn_prob < 0.65:
                st.warning(f"## 🟡 ความเสี่ยงปานกลาง ({churn_prob:.1%})")
            else:
                st.error(f"## ⚠️ ความเสี่ยงสูง ({churn_prob:.1%})")
            st.progress(churn_prob)
            
        with res_col2:
            st.write("**ปัจจัยที่มีผลต่อการทำนาย (+ เพิ่มเสี่ยง / - ลดเสี่ยง)**")
            chart_df = pd.DataFrame.from_dict(impacts, orient='index', columns=['Impact Weight'])
            st.bar_chart(chart_df)

        # ── Actionable Insights ──
        st.markdown("---")
        st.subheader("💡 คำแนะนำเชิงกลยุทธ์")
        if last_active > 14:
            st.info("📢 **Action:** ลูกค้าเริ่มนิ่งเงียบ ควรให้ฝ่าย Success ส่งอีเมล Re-engagement ทันที")
        if avg_sat < 3.0:
            st.error("🚨 **Alert:** คะแนนความพึงพอใจต่ำมาก ต้องรีบแก้ไขปัญหาทางเทคนิคหรือ Support ด่วน")
        if churn_prob > 0.65:
            st.warning("💸 **Retention:** มีความเสี่ยงสูงที่จะย้ายค่าย ควรพิจารณาเสนอส่วนลดหรือขยายเวลาใช้งานฟรี")
        elif churn_prob < 0.35:
            st.success("🌟 **Upsell:** ลูกค้ามีความสุขดี เป็นโอกาสเหมาะที่จะเสนออัปเกรดเป็นแพ็กเกจ Enterprise")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")

st.divider()
st.caption("AI Model: Gradient Boosting | Business Logic Layer: Enabled")
