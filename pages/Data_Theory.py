import streamlit as st

st.set_page_config(layout="centered")
st.title("📊 การเตรียมข้อมูลและแนวทางการพัฒนาโมเดล")

st.header("1️⃣ การเตรียมข้อมูล (Data Preparation)")
st.write("เริ่มโดยการหาหัวข้อที่เราสนใจ เมื่อได้หัวข้อแล้ว จากนั้นก็ รวบรวมหาข้อมูล dataset จากเว็บไซต์ต่างๆ ซึ่งข้อมูลที่ใช้ในการพัฒนาโมเดล คือ **ชุดข้อมูลที่เกี่ยวข้องกับโรคเบาหวาน** และ **ชุดข้อมูลผลไม้ที่มีภาพของผลไม้**แต่ละชนิด รวมถึงข้อมูลเมตา (CSV) ที่เก็บชื่อไฟล์และป้ายกำกับ (Label) ของผลไม้ โดยมีการตรวจสอบความสมบูรณ์ของข้อมูล และปรับแต่งข้อมูลให้เหมาะสม")
st.write("""
ในการพัฒนาโมเดล **Machine Learning** และ **Neural Network** นั้น ขั้นตอนแรกที่สำคัญที่สุดคือการเตรียมข้อมูล ซึ่งมีขั้นตอนต่างๆ ดังนี้:

1. **การทำความสะอาดข้อมูล (Data Cleaning)**:
    - ตรวจสอบข้อมูลที่ขาดหายหรือมีความผิดปกติ และ ทำการเติมค่าหรือกำจัดข้อมูลที่ไม่สมบูรณ์ออกจากชุดข้อมูล
    - ตรวจสอบค่าผิดปกติที่อาจเกิดขึ้น เช่น ค่าผิดปกติที่เป็น outliers

2. **การแปลงข้อมูล (Data Transformation)**:
    - การปรับสเกลข้อมูล (Feature Scaling) โดยใช้เทคนิคเช่น **StandardScaler** หรือ **MinMaxScaler** เพื่อให้ข้อมูลอยู่ในช่วงที่เหมาะสม
    - การแปลงข้อมูลเป็นตัวเลข (เช่น การแปลงคอลัมน์ที่เป็นข้อความให้เป็นตัวเลข เช่น **One-hot encoding**)

3. **การแยกข้อมูล (Data Splitting)**:
    - แยกข้อมูลออกเป็น **Training Set** และ **Test Set** โดยปกติจะใช้การแบ่งข้อมูลที่อัตราส่วน 80/20 หรือ 70/30 เพื่อใช้ในการฝึกโมเดลและทดสอบโมเดล
""")
st.write("ที่มา Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) และ [Fruit_Dataset](https://www.kaggle.com/datasets?search=fruit&fileType=csv&page=2)")

st.subheader("🔹 Feature ของ Dataset")
st.write("ชุดข้อมูลของโมเดล Machine Learning Model ประกอบไปด้วยตัวแปรหลักที่ใช้ในการพยากรณ์เบาหวาน เช่น:")
st.markdown("""
- **Age**: อายุของผู้ป่วย
- **Pregnancies**: จำนวนครั้งที่ตั้งครรภ์
- **BMI**: ค่าดัชนีมวลกาย
- **Glucose**: ระดับน้ำตาลในเลือด
- **BloodPressure**: ค่าความดันโลหิต
- **HbA1c**: ค่าระดับน้ำตาลสะสมในเลือด
- **LDL / HDL / Triglycerides**: ค่าตรวจไขมันในเลือด
- **FamilyHistory**: มีประวัติเบาหวานในครอบครัวหรือไม่
- **Hypertension**: มีภาวะความดันโลหิตสูงหรือไม่
- **Outcome**: ค่าผลลัพธ์ (0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน)
""")

st.write("ส่วนชุดข้อมูลของ Nueral Network Model ประกอบไปด้วย")
st.markdown("""
- **Fruit Dataset**: ชุดข้อมูลที่ประกอบไปด้วยภาพของผลไม้แต่ละชนิด รวมถึงชื่อไฟล์และป้ายกำกับ (Label) ของผลไม้    
            """)

st.header("2️⃣ ทฤษฎีของอัลกอริทึมที่พัฒนา")
st.write("ในการพัฒนาโมเดล AI ใช้อัลกอริทึม 2 ประเภทคือ **Machine Learning และ Neural Network**")

st.subheader("📌 Machine Learning: Logistic Regression")
st.write("Logistic Regression เป็นอัลกอริทึมสำหรับปัญหาจำแนกประเภท (Classification) มันทำงานโดยการประมาณค่าความน่าจะเป็นว่าเหตุการณ์จะเกิดขึ้นหรือไม่ (เช่น การทำนายว่าเป็นเบาหวานหรือไม่) โดยใช้ฟังก์ชัน Sigmoid เพื่อให้ผลลัพธ์อยู่ในช่วง [0, 1]")
st.latex(r'P(y=1|X) = \frac{1}{1 + e^{- (\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n)}}')

st.subheader("📌 Neural Network: Deep Learning")
st.write("Neural Network เป็นโมเดลที่ประกอบด้วยหลายชั้นของ neuron ที่เรียนรู้รูปแบบข้อมูลโดยใช้การคำนวณแบบ feed-forward และ backpropagation")
st.markdown("""เราสร้างโมเดล Convolutional Neural Network (CNN) ซึ่งประกอบด้วย
1. **Convolutional Layers**: ใช้ในการตรวจจับคุณสมบัติของภาพ เช่น ขอบและรูปทรง
2. **Pooling Layers**: ลดขนาดของข้อมูลและคำนึงถึงคุณสมบัติที่สำคัญที่สุด
3. **Fully Connected Layers**: ใช้เพื่อเชื่อมโยงข้อมูลจากเลเยอร์ก่อนหน้าไปยังการทำนายผล
4. **Sigmoid Activation**: ใช้ใน **output layer** เพื่อทำนายผลหลายคลาสพร้อมกัน (Multi-Label)""")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/800px-Artificial_neural_network.svg.png", caption="โครงสร้าง Neural Network", use_column_width=True )

st.header("3️⃣ ขั้นตอนการพัฒนาโมเดล")
st.write("กระบวนการพัฒนาโมเดลประกอบไปด้วยขั้นตอนต่อไปนี้:")
st.markdown("""
1. **Data Preprocessing**: ทำความสะอาดข้อมูล และปรับสเกลข้อมูล (Standardization)
2. **แบ่งชุดข้อมูล**: แบ่งเป็น Training Set และ Test Set (80:20)
3. **สร้างและเทรนโมเดล**
   - ใช้ Logistic Regression สำหรับ Machine Learning
   - ใช้ Neural Network (เช่น TensorFlow/Keras) สำหรับ Deep Learning
4. **ประเมินผลโมเดล**: คำนวณ Accuracy, Precision, Recall, และ F1-score
5. **นำโมเดลไปใช้กับ Streamlit**: เชื่อมโมเดลกับเว็บแอปพลิเคชัน
""")

st.success("🎯 การพัฒนาโมเดลเสร็จสมบูรณ์และพร้อมสำหรับการทดสอบ!")

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