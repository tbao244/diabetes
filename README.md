# 🏥 HealthAI — Hệ Thống Dự Đoán Tiểu Đường

Ứng dụng Streamlit đa trang với giao diện **Luxury Medical Dark** (Navy + Teal).

## Cấu trúc thư mục

```
diabetes_app/
│
├── main.py                  ← Entry point 
│
├── dataset/
│   └── diabetes.csv         ← Dataset tải về từ Kaggle
│ 
├── notebooks/
│   └── model_training       ← Train model
│
├── models/
│   ├── diabetes_model.h5    ← Mô hình Keras đã train
│   └── scaler.pkl           ← StandardScaler đã fit
│
├── pages/
│   ├── __init__.py
│   ├── login.py             ← Trang đăng nhập
│   ├── home.py              ← Trang chủ + gallery ảnh
│   └── predict.py           ← Form nhập liệu + popup kết quả
│
├── assets/
│   ├── style.css            ← Toàn bộ CSS/thiết kế chung
│   ├── anh_1.jpg            ← (tuỳ chọn) Ảnh thực tế cho gallery
│   ├── anh_2.jpg
│   └── anh_n.jpg            ← Thêm bao nhiêu ảnh cũng được
│ 
└── requirements.txt         ← Thư viện cần cài đặt    
```

## Cách chạy

```bash
# Cài dependencies
pip install -r requirements.txt
# Chạy ứng dụng
streamlit run main.py
```

## Tài khoản demo

| Email                  | Mật khẩu     | Tên            |
|------------------------|--------------|----------------|
| tonngokhong@health.ai  | tonngokhong  | Tôn Ngộ Không  |
| trubacgioi@health.ai   | trubatgioi   | Trư Bát Giới   |
| duongtang@health.ai    | duongtang123 | Đường Tăng     |
| satang@health.ai       | satang123    | Sa Tăng        |

## Thêm ảnh vào gallery

Đặt file ảnh vào thư mục `assets/` với tên `anh_1.jpg`, `anh_2.jpg`, … `anh_n.jpg`  
(hỗ trợ `.jpg`, `.jpeg`, `.png`, `.webp`).

Nếu không có ảnh thực, hệ thống tự động hiển thị **6 ảnh SVG minh hoạ** chủ đề y tế.

## Luồng điều hướng

```
Login page  ─(đăng nhập thành công)→  Home page
Home page   ─(bấm "Bắt đầu kiểm tra")→  Predict page
Predict page ─(bấm "Phân tích")→  Popup kết quả (overlay)
Predict page ─(bấm "Quay về")→  Home page
Home page   ─(bấm "Đăng xuất")→  Login page
```

## Ghi chú
- Kết quả AI hiển thị dưới dạng **popup overlay** với thanh tiến trình nguy cơ.
- CSS hoàn toàn override giao diện mặc định của Streamlit.
