import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ระบบวิเคราะห์ความเสี่ยงการยกเลิกบริการ",
    page_icon="📊",
    layout="wide", # ปรับเป็น wide เพื่อให้ดูเป็น Dashboard มากขึ้น
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

def adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat, usage_trend, last_active):
    """ฟังก์ชันปรับจูนความเสี่ยงตาม Business Logic"""
    factors = {}
    
    # 1. ระยะเวลาใช้งาน
    if tenure_days >= 730: factors['tenure'] = -0.25
    elif tenure_days < 100: factors['tenure'] = +0.10
    else: factors['tenure'] = 0

    # 2. ระดับแพ็กเกจ
    tier_map = {"enterprise": -0.20, "premium": -0.10, "standard": 0, "basic": +0.10}
    factors['tier'] = tier_map.get(plan_tier, 0)

    # 3. ความพึงพอใจ (ให้น้ำหนักสูงสุด)
    if avg_sat >= 4.5: factors['sat'] = -0.30
    elif avg_sat <= 2.0: factors['sat'] = +0.25
    else: factors['sat'] = 0

    # 4. พฤติกรรมล่าสุด (เพิ่มใหม่)
    if last_active > 30: factors['active'] = +0.20 # ไม่ใช้งานเกินเดือน = เสี่ยงมาก
    else: factors['active'] = -0.05

    total_adj = sum(factors.values())
    adjusted = raw_prob + total_adj
    return float(np.clip(adjusted, 0.01, 0.99)), factors

# ── UI Header ──────────────────────────────────────────────────────
st.title("📊 Churn Risk Analytics Dashboard")
st.markdown("ระบบทำนายความเสี่ยงการยกเลิกบริการด้วย Machine Learning")
st.divider()

# ── Sidebar สำหรับข้อมูลพื้นฐาน ──────────────────────────────────────
with st.sidebar:
    st.header("📍 ข้อมูลทั่วไป")
    industry = st.selectbox("อุตสาหกรรม", ["Cybersecurity", "DevTools", "EdTech", "FinTech", "HealthTech"])
    country = st.selectbox("ประเทศ", ["AU", "CA", "DE", "FR", "IN", "UK", "US"])
    referral = st.selectbox("ช่องทางที่มา", ["organic", "paid", "referral"])

# ── Main Input ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 ข้อมูลการใช้งาน")
    tenure_days = st.number_input("ระยะเวลาที่เป็นลูกค้า (วัน)", value=730, step=30)
    plan_tier = st.selectbox("ระดับแพ็กเกจ", ["basic", "standard", "premium", "enterprise"], index=3)
    last_active = st.slider("ไม่ได้ใช้งานมาแล้ว (วัน)", 0, 90, 5, help="สำคัญ: ลูกค้าที่หายไปนานมีความเสี่ยงสูง")
    usage_trend = st.slider("แนวโน้มการใช้งาน (Trend)", -1.0, 1.0, 0.2, step=0.1)

with col2:
    st.subheader("⭐ ความพึงพอใจ & บริการ")
    avg_sat = st.slider("ความพึงพอใจเฉลี่ย (1–5)", 1.0, 5.0, 4.5, step=0.1)
    escalation_rate = st.slider("สัดส่วน Ticket ที่บานปลาย (%)", 0.0, 1.0, 0.05)
    has_churn_history = st.checkbox("เคยขอยกเลิกบริการมาก่อน")
    n_churn_events = st.number_input("จำนวนครั้งที่เคยยกเลิก", 0, 10, 0) if has_churn_history else 0

st.divider()
if st.button("🔍 วิเคราะห์ความเสี่ยง", type="primary", use_container_width=True):
    # Prepare Data
    input_df = pd.DataFrame([{
        "tenure_days": tenure_days, "avg_sat": avg_sat, "avg_resolution": 12.0,
        "escalation_rate": escalation_rate, "annual_ratio": 1.1, "n_urgent": 1,
        "beta_ratio": 0.05, "sub_churn_rate": 0.05, "error_rate": 0.01,
        "n_trials": 1, "n_subs": 5, "n_escalated": 2, "n_tickets": 10,
        "has_churn_history": 1 if has_churn_history else 0, "n_churn_events": n_churn_events,
        "avg_seats": 10.0, "avg_first_resp": 3.0, "sat_std": 0.5,
        "usage_trend": usage_trend, "usage_days": int(tenure_days * 0.6),
        "pct_no_sat": 0.1, "ticket_acceleration": 0.0, "recent_escalations": 0,
        "min_sat": max(1.0, avg_sat - 1.0), "referral_source": referral,
        "plan_tier": plan_tier, "industry": industry, "country": country
    }])

    try:
        raw_prob = model.predict_proba(input_df)[0][1]
        churn_prob, factors = adjust_probability(raw_prob, tenure_days, plan_tier, avg_sat, usage_trend, last_active)

        # ── การแสดงผลลัพธ์ ──
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Probability of Churn", f"{churn_prob:.1%}")
            if churn_prob < 0.3:
                st.success("### ✅ ความเสี่ยงต่ำ")
            elif churn_prob < 0.6:
                st.warning("### 🟡 ความเสี่ยงปานกลาง")
            else:
                st.error("### ⚠️ ความเสี่ยงสูง")
            
            st.progress(churn_prob)

        with res_col2:
            # กราฟแสดงผลกระทบของปัจจัย (Explainability)
            labels = list(factors.keys())
            values = list(factors.values())
            colors = ['red' if v > 0 else 'green' for v in values]
            
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation='h',
                marker_color=colors
            ))
            fig.update_layout(title="ปัจจัยที่ส่งผลต่อการทำนาย (+ เพิ่มเสี่ยง, - ลดเสี่ยง)", height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # ── คำแนะนำเชิงลึก ──
        st.subheader("💡 คำแนะนำเชิงกลยุทธ์ (Actionable Insights)")
        tips = []
        if last_active > 14: tips.append("- **Re-engagement:** ลูกค้าไม่ได้เข้าใช้นานเกิน 2 สัปดาห์ ควรส่งอีเมลแนะนำฟีเจอร์ใหม่")
        if avg_sat < 3.5: tips.append("- **Health Check:** คะแนนความพึงพอใจต่ำกว่ามาตรฐาน ควรให้ฝ่ายสนับสนุนติดต่อสอบถามปัญหา")
        if churn_prob > 0.6: tips.append("- **Retention Offer:** เสี่ยงสูงมาก ควรเสนอส่วนลดหรือแพ็กเกจพิเศษเพื่อดึงตัวไว้")
        
        if not tips: tips.append("- รักษามาตรฐานการบริการที่ดีเยี่ยมต่อไป และพิจารณาทำ Case Study ร่วมกับลูกค้า")
        
        for tip in tips:
            st.markdown(tip)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

st.divider()
st.caption("AI Model: Gradient Boosting Classifier | Threshold: Dynamic Adjustment")
