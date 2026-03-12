RavenStack Churn Prediction Project 🚀
โปรเจกต์นี้เป็นการนำเทคนิค Machine Learning มาใช้ในการทำนายการเลิกใช้งานของลูกค้า (Customer Churn Prediction) สำหรับแพลตฟอร์ม RavenStack โดยใช้ข้อมูลการใช้งานจริง (Synthetic Data) เพื่อช่วยให้ธุรกิจสามารถระบุลูกค้าที่มีความเสี่ยงและหาแนวทางรักษาลูกค้าไว้ได้ล่วงหน้า

📊 ข้อมูลโปรเจกต์ (Dataset Overview)
ข้อมูลที่ใช้ประกอบด้วย 5 ส่วนหลัก ได้แก่:

Accounts: ข้อมูลบัญชีผู้ใช้พื้นฐาน (500 รายการ)

Subscriptions: ประวัติการสมัครแพลนต่างๆ (5,000 รายการ)

Feature Usage: พฤติกรรมการใช้งานฟีเจอร์ในระบบ (25,000 รายการ)

Support Tickets: ประวัติการติดต่อเจ้าหน้าที่ (2,000 รายการ)

Churn Events: บันทึกการเลิกใช้งานจริง (600 รายการ)

🛠️ ขั้นตอนการดำเนินงาน (Project Workflow)
1. Data Loading & EDA
โหลดข้อมูลและตรวจสอบคุณภาพของข้อมูล

ทำ Exploratory Data Analysis (EDA) เพื่อหาความสัมพันธ์ของตัวแปร เช่น อัตราการ Churn รายอุตสาหกรรม และช่องทางการแนะนำลูกค้า

Insight สำคัญ: พบว่าฟีเจอร์พื้นฐานบางอย่างมี Signal ที่อ่อนมาก จึงเน้นไปที่ฟีเจอร์ที่แสดงพฤติกรรมการใช้งานจริง

2. Feature Engineering
สร้างฟีเจอร์ใหม่จากข้อมูลพฤติกรรม เช่น usage_frequency, ticket_count, days_since_last_use

คำนวณความเสี่ยงของลูกค้า (Risk Scores)

3. Model Training & Evaluation
ได้มีการทดลองใช้โมเดลหลายประเภทเพื่อเปรียบเทียบประสิทธิภาพ:

Logistic Regression: เป็นโมเดล Baseline

Random Forest: เพื่อหา Feature Importance

Gradient Boosting: เพื่อเพิ่มความแม่นยำ

Voting Classifier (Ensemble): รวมจุดแข็งของโมเดลต่างๆ เข้าด้วยกัน

4. Results & Reporting
วิเคราะห์ประสิทธิภาพด้วย Metrics: Precision, Recall, F1-Score

บันทึกคะแนนความเสี่ยงของลูกค้าแต่ละรายลงในไฟล์ customer_risk_scores.csv เพื่อนำไปใช้ต่อในทีมธุรกิจ

📈 ผลลัพธ์ที่ได้
โมเดลที่เลือกใช้ (Best Model) สามารถทำนายการ Churn ได้อย่างแม่นยำ พร้อมทั้งสามารถระบุได้ว่าฟีเจอร์ใดเป็นปัจจัยหลักที่ส่งผลต่อการตัดสินใจของลูกค้า

🚀 วิธีการใช้งาน
Clone repository นี้ไปยังเครื่องของคุณ

ติดตั้ง library ที่จำเป็น: pip install pandas numpy matplotlib seaborn scikit-learn joblib

รันไฟล์ model2.ipynb เพื่อดูขั้นตอนการวิเคราะห์และฝึกสอนโมเดล
