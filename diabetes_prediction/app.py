import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

import os
# 1. Lấy thư mục chứa file app.py (tức là thư mục 'diabetes_prediction')
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Lùi ra ngoài 1 cấp để về thư mục gốc của Repository ('diabetes')
REPO_ROOT = os.path.dirname(APP_DIR)
# 3. Trỏ chính xác vào thư mục models nằm ở gốc Repo
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'diabetes_model.h5')
SCALER_PATH = os.path.join(REPO_ROOT, 'models', 'scaler.pkl')
# 4. Tiến hành nạp mô hình và bắt lỗi trực quan để debug nếu có sự cố
try:
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"❌ Hệ thống không tìm thấy file bảo bối!")
    st.info(f"Chi tiết lỗi hệ thống: {e}")
    st.warning(f"Đường dẫn hệ thống đang tìm thực tế là: {MODEL_PATH}")
    # In ra danh sách file ở thư mục gốc để kiểm tra Git đã nhận file chưa
    st.write("Các thư mục hiện có tại gốc Repo:", os.listdir(REPO_ROOT))

st.set_page_config(
    page_title="Dự đoán & Tư vấn Bệnh Tiểu đường",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource  
def load_assets():
    model = load_model('models/diabetes_model.h5') 
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Không tìm thấy file mô hình hoặc scaler trong thư mục 'models/'. Chi tiết lỗi: {e}")
    st.stop()

st.title("Hệ Thống AI Dự Đoán Diabetes & Tư Vấn Sức Khỏe")
st.write("Nhập các chỉ số sinh tồn bên dưới để kiểm tra nguy cơ sức khỏe dựa trên mô hình mạng nơ-ron.")

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📋 Nhập thông số lâm sàng")
    pregnancies = st.number_input("Số lần mang thai (Pregnancies):", min_value=0, max_value=20, value=1, step=1)
    glucose = st.slider("Chỉ số Đường huyết (Glucose):", min_value=0, max_value=200, value=120)
    blood_pressure = st.slider("Huyết áp tâm trương (BloodPressure) - mmHg:", min_value=0, max_value=130, value=70)
    skin_thickness = st.slider("Độ dày nếp gấp da cơ đầu tam đầu (SkinThickness) - mm:", min_value=0, max_value=100, value=20)
    insulin = st.slider("Chỉ số Insulin - mu U/ml:", min_value=0, max_value=900, value=80)
    weight = st.number_input("Cân nặng (Weight) - kg:", min_value=1.0, max_value=200.0, value=70.0, step=0.5)
    height_cm = st.number_input("Chiều cao (Height) - cm:", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
    dpf = st.number_input("Chức năng phả hệ tiểu đường (Diabetes Pedigree Function):", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Tuổi (Age):", min_value=1, max_value=120, value=30, step=1)
    predict_btn = st.button("🚀 Bắt đầu dự đoán", type="primary")

    height_m = height_cm / 100.0
    if height_m > 0:
        bmi = weight / (height_m ** 2)
    else:
        bmi = 0.0

with col2:
    st.subheader("📊 Kết quả phân tích từ AI")
    
    if predict_btn:
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        user_data_scaled = scaler.transform(user_data)
        prediction_prob = model.predict(user_data_scaled)[0][0]
        st.write(f"**Xác suất mắc bệnh cấu thành:** `{prediction_prob * 100:.2f}%`")
        if prediction_prob < 0.3:
            st.success(f"🟢 **Mức độ: NGUY CƠ THẤP** (Xác suất mắc bệnh: {prediction_prob * 100:.2f}%)")
            st.markdown("""
            **📋 Lời khuyên sức khỏe:**
            * **Duy trì phong độ:** Chỉ số sinh tồn của bạn đang ở trạng thái lý tưởng. Hãy tiếp tục duy trì chế độ ăn uống cân bằng hiện tại.
            * **Hoạt động thể chất:** Khuyến khích duy trì thể dục ít nhất 150 phút/tuần (chạy bộ, bơi lội, đạp xe) để giữ mức cân nặng và chỉ số BMI ổn định.
            * **Kiểm tra định kỳ:** Dù nguy cơ thấp, bạn vẫn nên chủ động kiểm tra lượng Đường huyết (Glucose) trong các kỳ khám sức khỏe tổng quát hàng năm.
            """)
        elif 0.3 <= prediction_prob < 0.7:
            st.warning(f"🟡 **Mức độ: NGUY CƠ TRUNG BÌNH** (Xác suất mắc bệnh: {prediction_prob * 100:.2f}%)")
            st.markdown("""
            **📋 Lời khuyên sức khỏe:**
            * **Cảnh báo sớm:** Các chỉ số của bạn đang nằm trong vùng tiền tiểu đường hoặc có dấu hiệu kháng insulin nhẹ. Cần chủ động thay đổi lối sống ngay từ bây giờ để đảo ngược nguy cơ.
            * **Điều chỉnh chế độ ăn:**
                * Hạn chế tối đa carbohydrate tinh chế (nước ngọt, bánh kẹo, tinh bột trắng).
                * Bổ sung thêm nhiều chất xơ từ rau xanh và các loại hạt nguyên cám để làm chậm quá trình hấp thụ đường sau ăn.
            * **Kiểm soát cân nặng:** Nếu chỉ số BMI hiện tại của bạn đang ở mức thừa cân, hãy lên kế hoạch giảm từ 5% - 7% trọng lượng cơ thể để cải thiện độ nhạy insulin.
            """)
        else:
            st.error(f"🔴 **Mức độ: NGUY CƠ CAO** (Xác suất mắc bệnh: {prediction_prob * 100:.2f}%)")
            st.markdown("**📋Chế độ dinh dưỡng nghiêm ngặt:**")
            st.info("""
            * **Diet plays an extremely important role** in helping diabetics control their disease.
            * **Basic principles in diabetic diet:** * Ensure adequate nutrition.
                * Do not increase blood sugar much after meals.
                * Do not lower blood sugar far from meals to maintain normal physical activity and maintain reasonable weight.
            * **Design meals** that are simple, not too expensive and suitable for local customs.
            * **Balance the ratio** of carbohydrates, proteins and fats; supplement foods rich in nutrients, low in fat and calories such as vegetables, fruits, whole grains; monitor blood sugar after meals...
            * ⚠️ **Important Note:** Patients can **consult a doctor or nutritionist** as soon as possible for professional guidance on a suitable clinical diet.
            """)
    else:
        st.info("💡 Điền đầy đủ thông tin ở cột bên trái và bấm nút **Bắt đầu dự đoán** để xem kết quả phân tích phân loại.")