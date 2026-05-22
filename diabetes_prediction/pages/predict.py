import streamlit as st
import numpy as np
import os
import plotly.graph_objects as go

_model  = None
_scaler = None

def _load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def _load_model_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        import pickle
        try:
            from tensorflow.keras.models import load_model
            km_load = load_model
        except ImportError:
            try:
                from keras.models import load_model
                km_load = load_model
            except ImportError as e:
                st.error(f" Không thể import Keras/TensorFlow. Lỗi: {e}")
                return None, None

        base = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base, "models", "diabetes_model.h5")
        scaler_path = os.path.join(base, "models", "scaler.pkl")

        if not os.path.exists(model_path):
            st.error(f" Không tìm thấy file mô hình tại: `{model_path}`")
            return None, None
        if not os.path.exists(scaler_path):
            st.error(f" Không tìm thấy scaler tại: `{scaler_path}`")
            return None, None

        try:
            _model = km_load(model_path)
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
        except Exception as e:
            st.error(f"Lỗi tải mô hình: {e}")
            return None, None

    return _model, _scaler

def _risk(prob):
    if prob < 0.3:  
        return "#10b981", "#d1fae5", "NGUY CƠ THẤP", "✅", "Thấp"
    if prob < 0.7:  
        return "#f59e0b", "#fef3c7", "NGUY CƠ TRUNG BÌNH", "⚠️", "Trung bình"
    return "#ef4444", "#fee2e2", "NGUY CƠ CAO", "🚨", "Cao"

def create_gauge_chart(probability):
    if probability < 0.3:
        color = "#10b981"  
    elif probability < 0.7:
        color = "#f59e0b"  
    else:
        color = "#ef4444" 

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {
            'suffix': "%", 
            'font': {'size': 36, 'color': color, 'weight': 'bold'}
        },
        title = {
            'text': "Mức độ rủi ro", 
            'font': {'size': 14, 'color': '#64748b'}
        },
        gauge = {
            'axis': {
                'range': [0, 100], 
                'tickwidth': 1, 
                'tickcolor': "#cbd5e1",
                'tickfont': {'color': '#94a3b8'}
            },
            'bar': {'color': color, 'thickness': 0.75}, 
            'bgcolor': "rgba(0,0,0,0)", 
            'borderwidth': 0, 
            'steps': [
                {'range': [0, 30], 'color': "#d1fae5"}, 
                {'range': [30, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#fee2e2"} 
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.85,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=260, 
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",  
        font={'family': "Lora, serif"} 
    )
    
    return fig

def show():
    _load_css()

    user = st.session_state.get("user_name", "Người dùng")
    initial = user[0].upper() if user else "U"

    # Topnav
    st.markdown(f"""
    <div class="topnav" style="box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <div class="topnav-brand">🏥 Health<span style="color:#2563eb;">AI</span></div>
        <div class="topnav-user">
            <span style="color:#334155; font-weight:600;">{user}</span>
            <div class="topnav-avatar" style="background:#2563eb; color:white;">{initial}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("show_result", False):
        # HIỂN THỊ KẾT QUẢ
        result = st.session_state["result_data"]
        
        st.markdown("""
        <div style="background:linear-gradient(135deg,#e0f2fe,#f0fdf4); border-radius: 15px;
             border:1px solid #e2e8f0; padding:32px 24px; text-align:center; margin-bottom: 25px; margin-top: 15px;">
            <div style="font-family:'Lora',serif;font-size:28px;font-weight:700;color:#0f172a;">
                📊 Hồ Sơ Phân Tích Sức Khỏe
            </div>
            <p style="font-size:14px;color:#475569;margin-top:8px;">
                Báo cáo đánh giá nguy cơ tiểu đường dựa trên dữ liệu lâm sàng
            </p>
        </div>
        """, unsafe_allow_html=True)
        
      
        _, main_col, _ = st.columns([1, 8, 1])
        with main_col:
            col1, col2 = st.columns([1.1, 0.9], gap="large")
            
            with col1:
                st.markdown(f"""
                <div style="background:{result['bg_col']}; border: 1px solid {result['color']}40; border-radius:24px; padding:35px 25px; text-align:center; box-shadow: 0 10px 25px {result['color']}20;">
                    <div style="font-size:72px; line-height: 1;">{result['emoji']}</div>
                    <div style="font-size:32px; font-family:'Lora',serif; font-weight:800; color:{result['color']}; margin:15px 0 5px 0;">
                        {result['level_txt']}
                    </div>
                    <div style="font-size:14px; font-weight: 500; color:#64748b; text-transform: uppercase; letter-spacing: 1px;">Xác suất mắc bệnh</div>
                </div>
                """, unsafe_allow_html=True)
                
            
                fig = create_gauge_chart(result['pct'] / 100) 
                st.plotly_chart(fig, use_container_width=True)

            with col2:
        
                st.markdown(f"""
                <div style="background:white; border:1px solid #e2e8f0; border-radius:24px; padding:30px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); height: 100%;">
                    <div style="font-family:'Lora',serif; font-size:20px; font-weight:700; margin-bottom:20px; color:#0f172a; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;">
                        📋 Thông số đầu vào
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">⚖️</span><br>
                            <span style="color:#64748b; font-size: 12px;">BMI</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['bmi']:.1f}</b>
                        </div>
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">🩸</span><br>
                            <span style="color:#64748b; font-size: 12px;">Glucose</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['glucose']}</b>
                        </div>
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">🫀</span><br>
                            <span style="color:#64748b; font-size: 12px;">Huyết áp</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['bp']}</b>
                        </div>
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">💉</span><br>
                            <span style="color:#64748b; font-size: 12px;">Insulin</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['insulin']}</b>
                        </div>
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">🎂</span><br>
                            <span style="color:#64748b; font-size: 12px;">Tuổi</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['age']}</b>
                        </div>
                        <div style="background:#f8fafc; padding:12px; border-radius:10px; border:1px solid #f1f5f9;">
                            <span style="font-size:20px;">🧬</span><br>
                            <span style="color:#64748b; font-size: 12px;">Chức năng DPF</span><br>
                            <b style="font-size: 18px; color:#0f172a;">{result['dpf']:.2f}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
      
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='font-family: Lora, serif; color: {result['color']};'>💡 Kế hoạch hành động đề xuất</h3>", unsafe_allow_html=True)
            
            advice_items = result['advice_items']
            for title, desc in advice_items:
                with st.expander(title, expanded=True):
                    st.write(desc)
            
      
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🔄 Thực hiện đánh giá mới", type="primary", use_container_width=True):
                    st.session_state["show_result"] = False
                    st.session_state.pop("result_data", None)
                    st.rerun()
            
            _, back_col, _ = st.columns([1, 2, 1])
            with back_col:
                if st.button("🏠 Về trang chủ", use_container_width=True):
                    st.session_state["page"] = "home"
                    st.session_state["show_result"] = False
                    st.rerun()
                    
            st.markdown("""
            <hr style="margin-top: 50px;">
            <p style="text-align: center; font-size: 12px; color: #94a3b8;">
                ⚠️ <b>Miễn trừ trách nhiệm:</b> Hệ thống AI này chỉ cung cấp thông tin mang tính chất tham khảo dựa trên dữ liệu thống kê. <br>Đây không phải là chẩn đoán y khoa chính thức. Vui lòng tham vấn bác sĩ chuyên khoa để có kết luận chính xác nhất.
            </p>
            """, unsafe_allow_html=True)
    
    else:
  
        st.markdown("""
        <div style="background:linear-gradient(135deg,#e0f2fe,#f0fdf4); border-radius: 15px;
             border:1px solid #e2e8f0; padding:32px 24px; text-align:center; margin-bottom: 25px; margin-top: 15px;">
            <div style="font-family:'Lora',serif;font-size:28px;font-weight:700;color:#0f172a;">
                🩺 Đánh Giá Nguy Cơ Tiểu Đường
            </div>
            <p style="font-size:14px;color:#475569;margin-top:8px;">
                Nhập các chỉ số lâm sàng hiện tại của bạn. Hệ thống AI sẽ phân tích và đưa ra kết quả ngay lập tức.
            </p>
        </div>
        """, unsafe_allow_html=True)

        _, main_col, _ = st.columns([1, 10, 1])

        with main_col:
            left, right = st.columns(2, gap="large")

            with left:
                with st.container(border=True):
                    st.markdown("""
                    <div style="font-family:'Lora',serif; font-size:20px; font-weight:700; 
                         color:#0f172a; margin-bottom:15px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;">
                        📋 Bảng Thông Số Lâm Sàng
                    </div>
                    """, unsafe_allow_html=True)

                    pregnancies = st.number_input("🤰 Số lần mang thai", min_value=0, max_value=20, value=1, step=1)
                    glucose = st.slider("🩸 Đường huyết (mg/dL)", 0, 200, 120, help="Nồng độ glucose huyết tương trong thử nghiệm dung nạp glucose đường uống trong 2 giờ")
                    blood_pressure = st.slider("🫀 Huyết áp tâm trương (mmHg)", 0, 130, 70)
                    skin_thickness = st.slider("🤏 Độ dày nếp gấp da cơ tam đầu (mm)", 0, 100, 20)
                    insulin = st.slider("💉 Insulin huyết thanh 2 giờ (μU/mL)", 0, 900, 80)

            with right:
                with st.container(border=True):
                    st.markdown("""
                    <div style="font-family:'Lora',serif; font-size:20px; font-weight:700; 
                         color:#0f172a; margin-bottom:15px; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;">
                        📏 Nhân Trắc Học & Tiền Sử
                    </div>
                    """, unsafe_allow_html=True)

                    col_w, col_h = st.columns(2)
                    with col_w:
                        weight = st.number_input("⚖️ Cân nặng (kg)", min_value=1.0, max_value=200.0, value=70.0, step=0.5)
                    with col_h:
                        height_cm = st.number_input("📏 Chiều cao (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)

                    height_m = height_cm / 100.0
                    bmi = weight / (height_m ** 2) if height_m > 0 else 0.0
                    
                    if bmi < 18.5:
                        bmi_color, bmi_bg, bmi_label = "#f59e0b", "#fef3c7", "Thiếu cân"
                    elif bmi < 25:
                        bmi_color, bmi_bg, bmi_label = "#10b981", "#d1fae5", "Bình thường"
                    elif bmi < 30:
                        bmi_color, bmi_bg, bmi_label = "#f59e0b", "#fef3c7", "Thừa cân"
                    else:
                        bmi_color, bmi_bg, bmi_label = "#ef4444", "#fee2e2", "Béo phì"
                    
                    st.markdown(f"""
                    <div style="background:{bmi_bg}; border-radius:12px; padding:15px 20px; margin: 15px 0;
                         display:flex; align-items:center; justify-content:space-between; border: 1px solid {bmi_color}40;">
                        <div>
                            <div style="font-size:12px; color:#64748b; font-weight: 600; text-transform: uppercase;">Chỉ số BMI</div>
                            <div style="font-family:'Lora',serif; font-size:36px; font-weight:800; color:{bmi_color}; line-height: 1.2;">{bmi:.1f}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:15px; font-weight:700; color:{bmi_color}; background: white; padding: 4px 10px; border-radius: 20px; display: inline-block;">{bmi_label}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    dpf = st.number_input("🧬 Chức năng phả hệ tiểu đường (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Chỉ số đánh giá mức độ di truyền bệnh từ gia đình")
                    age = st.number_input("🎂 Tuổi", min_value=1, max_value=120, value=30, step=1)
                
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            _, btn_mid, _ = st.columns([1, 3, 1])
            with btn_mid:
                predict_btn = st.button("🚀 Bắt Đầu Phân Tích AI", type="primary", use_container_width=True)

            _, back_mid, _ = st.columns([1, 3, 1])
            with back_mid:
                if st.button("← Quay về trang chủ", use_container_width=True):
                    st.session_state["page"] = "home"
                    st.rerun()

            # Xử lý dự đoán
            if predict_btn:
                with st.spinner("🔬 AI đang phân tích hồ sơ của bạn..."):
                    model, scaler = _load_model_scaler()

                if model and scaler:
                    arr = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])
                    prob = float(model.predict(scaler.transform(arr))[0][0])
                    color, bg_col, level_txt, emoji, _ = _risk(prob)
                    pct = prob * 100

                 
                    if prob < 0.3:
                        advice_items = [
                            ("✅ Duy trì phong độ", "Chỉ số đang lý tưởng. Tiếp tục duy trì chế độ ăn uống cân bằng hiện tại."),
                            ("🏃 Hoạt động thể chất", "Duy trì ≥150 phút/tuần các hoạt động thể thao (chạy bộ, bơi lội, đạp xe)."),
                            ("🗓️ Kiểm tra định kỳ", "Đo đường huyết trong các kỳ khám sức khỏe tổng quát hàng năm để theo dõi."),
                        ]
                    elif prob < 0.7:
                        advice_items = [
                            ("⚠️ Cảnh báo sớm (Tiền tiểu đường)", "Chỉ số của bạn nằm trong vùng nguy cơ. Cần thay đổi lối sống ngay hôm nay."),
                            ("🥗 Điều chỉnh ăn uống", "Hạn chế tối đa tinh bột trắng, đồ ngọt. Tăng cường rau xanh, ngũ cốc nguyên cám và chất xơ."),
                            ("⚖️ Kiểm soát cân nặng", "Nghiên cứu cho thấy giảm 5–7% trọng lượng cơ thể giúp giảm đáng kể nguy cơ phát triển bệnh."),
                        ]
                    else:
                        advice_items = [
                            ("🚨 Cần tham vấn Y khoa ngay", "Mức nguy cơ của bạn đang ở mức RẤT CAO. Hãy đặt lịch hẹn với bác sĩ nội tiết sớm nhất có thể."),
                            ("🍽️ Kiểm soát đường huyết nghiêm ngặt", "Chia nhỏ bữa ăn, tuyệt đối không tiêu thụ đồ uống có đường để tránh đường huyết tăng vọt."),
                            ("📊 Bắt đầu theo dõi tại nhà", "Mua máy đo đường huyết cá nhân và bắt đầu ghi chép lại chỉ số sau các bữa ăn chính."),
                        ]

                    st.session_state["show_result"] = True
                    st.session_state["result_data"] = {
                        "pct": pct, "color": color, "bg_col": bg_col,
                        "level_txt": level_txt, "emoji": emoji,
                        "bmi": bmi, "glucose": glucose, "bp": blood_pressure,
                        "insulin": insulin, "age": age, "dpf": dpf,
                        "advice_items": advice_items
                    }
                    st.rerun()