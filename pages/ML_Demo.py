import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
st.set_page_config( layout="centered")
st.title("Machine Learning Demo")


# โหลดข้อมูล
df = pd.read_csv("diabetes_dataset.csv")

st.title("📊 Data Visualization")

# สร้างกราฟ Scatter Plot
fig, ax = plt.subplots()
ax.scatter(df[df["Outcome"] == 0]["Glucose"], df[df["Outcome"] == 0]["BMI"], label="Is'n Diabetes (0)", alpha=0.5)
ax.scatter(df[df["Outcome"] == 1]["Glucose"], df[df["Outcome"] == 1]["BMI"], label="Diabetes (1)", alpha=0.5, color="red")

ax.set_xlabel("Glucose Level")
ax.set_ylabel("BMI")
ax.set_title("Scatter Plot: Glucose vs BMI from Outcome")
ax.legend()
ax.grid(True)

# แสดงกราฟใน Streamlit
st.pyplot(fig)

# แยก Features และ Target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# แบ่งข้อมูลเป็น Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ปรับสเกลข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# เทรนโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# บันทึกโมเดลและ Scaler ใหม่
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ โมเดลและ Scaler ถูกบันทึกใหม่แล้ว!")

# ส่วนหัวของเว็บแอป
st.title("🔍 Diabetes Prediction App")
st.write("กรอกข้อมูลเพื่อทำนายโอกาสเป็นเบาหวาน หรือ กดปุ่ม **สุ่มค่า** ")

def predict_diabetes(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return ["เป็นเบาหวาน" if pred == 1 else "ไม่เป็นเบาหวาน" for pred in prediction]


columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 'LDL', 'HDL', 'Triglycerides',
           'WaistCircumference', 'HipCircumference', 'WHR', 'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse']

# การเก็บข้อมูลใน session_state เพื่อไม่ให้รีเซ็ต
if 'input_data' not in st.session_state:
    st.session_state.input_data = {col: None for col in columns}  # กำหนดค่าเริ่มต้น

def generate_default_value(col):
    default_values = {
        'Age': random.randint(20, 80),
        'Pregnancies': random.randint(0, 10),
        'BMI': round(random.uniform(18.5, 35.0), 2),
        'Glucose': round(random.uniform(70, 200), 1),
        'BloodPressure': round(random.uniform(60, 140), 1),
        'HbA1c': round(random.uniform(4.0, 10.0), 1),
        'LDL': round(random.uniform(50, 200), 1),
        'HDL': round(random.uniform(30, 80), 1),
        'Triglycerides': round(random.uniform(50, 250), 1),
        'WaistCircumference': round(random.uniform(60, 120), 1),
        'HipCircumference': round(random.uniform(80, 140), 1),
        'WHR': round(random.uniform(0.7, 1.2), 2),
        'FamilyHistory': random.choice([0, 1]),
        'DietType': random.choice([0, 1]),
        'Hypertension': random.choice([0, 1]),
        'MedicationUse': random.choice([0, 1])
    }
    return default_values.get(col, 0)

# ปุ่มในรูปแบบแนวนอน
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔮 สุ่มค่า"):
        for col in columns:
            st.session_state.input_data[col] = generate_default_value(col)

with col2:
    if st.button("🔄 รีเซ็ตข้อมูล"):
        # รีเซ็ตข้อมูลใน session_state
        st.session_state.input_data = {col: None for col in columns}

with col3:
    if st.button("🔮 ทำนายผล"):
        if None in st.session_state.input_data.values():
            st.warning("กรุณากรอกข้อมูลให้ครบถ้วน")
        else:
            input_df = pd.DataFrame([list(st.session_state.input_data.values())], columns=columns)
            prediction = predict_diabetes(input_df)[0]
            st.success(f"ผลลัพธ์: **{prediction}**")

# ถ้า User กรอกข้อมูลเอง
for col in columns:
    st.session_state.input_data[col] = st.number_input(f"{col}", value=st.session_state.input_data[col])



# ข้อมูลติดต่อ
st.sidebar.title("📞 ติดต่อ")
st.sidebar.info("""
    **ติดต่อผู้พัฒนาได้ที่**:

    **Email**: [s6604062663108@email.kmutnb](mailto:s6604062663108@email.kmutnb.ac.th)   
    **ชื่อ**: ดุลยวัต สีแก้ว              
    **รหัสนักศึกษา**: 6604062663108

    **Email**: [s6604062663159@email.kmutnb](mailto:s6604062663159@email.kmutnb.ac.th) 
    **ชื่อ**: ธีรภัทร จุลเวช             
    **รหัสนักศึกษา**: 6604062663159
""")