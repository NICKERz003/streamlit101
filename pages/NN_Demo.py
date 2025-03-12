import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt


st.title("Neural Network Demo")

# โหลดโมเดล
model = load_model('fruit_classifier_model.h5')
# ฟังก์ชั่นสำหรับแปลงภาพ
def prepare_image(image):
    image = image.resize((128, 128))  # ขนาดที่โมเดลใช้
    image = np.array(image) / 255.0  # นำภาพไปทำ normalization
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เป็น (1, 128, 128, 3)
    return image

# สร้างหน้าเว็บ
st.title("🍊 Fruit Classifier")
uploaded_file = st.file_uploader("Upload an image of fruit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # โหลดภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # แปลงภาพแล้วทำการทำนาย
    image_array = prepare_image(image)
    prediction = model.predict(image_array)

    # หา index ของคลาสที่มีค่าความน่าจะเป็นสูงที่สุด
    predicted_class = np.argmax(prediction, axis=1)

    # แสดงผลลัพธ์
    fruit_labels = ["Apple", "Banana", "Grapes", "Kiwi", "Mango", "Orange", "Pineapple", "Watermelon"]  # ใส่ชื่อผลไม้ตามลำดับ
    predicted_fruit = fruit_labels[predicted_class[0]]
    st.success(f"Prediction: {predicted_fruit}")


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