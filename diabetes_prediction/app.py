import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
try:
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Không tìm thấy file")
    st.info(f"Chi tiết lỗi: {e}")
    st.warning(f"Đường dẫn hệ thống đã tìm là: {MODEL_PATH}")

st.set_page_config(
    page_title="Dự đoán & Tư vấn Bệnh Tiểu đường",
    page_icon="🏥",
    layout="centered"
)

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
            st.markdown("**📋 Tư vấn y khoa (Chế độ dinh dưỡng nghiêm ngặt):**")
            st.info("""
            * **Chế độ ăn uống đóng vai trò cực kỳ quan trọng** trong việc giúp người bệnh tiểu đường kiểm soát tình trạng bệnh.
            * **Các nguyên tắc cơ bản trong chế độ ăn của người tiểu đường:**
                * Đảm bảo cung cấp đầy đủ chất dinh dưỡng cho cơ thể.
                * Không làm tăng nhanh lượng đường huyết một cách đột ngột sau khi ăn.
                * Không để đường huyết bị hạ quá thấp khi ở xa các bữa ăn nhằm duy trì các hoạt động thể chất bình thường và giữ mức cân nặng hợp lý.
            * **Thiết kế các bữa ăn** đơn giản, không quá tốn kém và phù hợp với phong tục, tập quán địa phương.
            * **Cân đối tỷ lệ** giữa các chất đường bột (carbohydrates), chất đạm (proteins) và chất béo (fats); tăng cường bổ sung các loại thực phẩm giàu dinh dưỡng nhưng ít béo và ít calo như rau xanh, trái cây, ngũ cốc nguyên hạt; chủ động theo dõi lượng đường huyết sau các bữa ăn...
            * ⚠️ **Lưu ý quan trọng:** Người bệnh nên **tham khảo ý kiến của bác sĩ hoặc chuyên gia dinh dưỡng** càng sớm càng tốt để nhận được hướng dẫn chuyên môn về một chế độ ăn lâm sàng phù hợp nhất.
            """)
    else:
        st.info("💡 Điền đầy đủ thông tin ở cột bên trái và bấm nút **Bắt đầu dự đoán** để xem kết quả phân tích phân loại.")