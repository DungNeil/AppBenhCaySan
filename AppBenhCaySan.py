import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="AppBenhCaySan - Nhận Diện Bệnh Sắn",
    page_icon="🍃",
    layout="centered"
)

st.title("🍃 AppBenhCaySan")
st.write("Hệ thống AI chẩn đoán bệnh lá sắn trực tuyến.")

# --- TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    # Cách này sẽ ép Keras nạp model mà không bắt bẻ các lớp tùy chỉnh
    model = tf.keras.models.load_model(
        'best_model_cassava.h5', 
        compile=False, 
        custom_objects=None
    )
    return model@st.cache_resource
def load_model():
    # Bỏ các tham số rườm rà, để Keras tự xử lý cấu trúc mới
    model = tf.keras.models.load_model('best_model_cassava.h5', compile=False)
    return model

with st.spinner('Đang tải bộ não AI...'):
    model = load_model()

# --- DANH SÁCH NHÃN (Dựa trên tên thư mục bạn đã đổi) ---
# Thứ tự trong list này PHẢI khớp với thứ tự bảng chữ cái của thư mục lúc bạn Train
class_names = [
    'Cassava Bacterial Blight (CBB) - Bệnh cháy lá do vi khuẩn',
    'Cassava Brown Streak Disease (CBSD) - Bệnh sọc nâu thân sắn',
    'Cassava Green Mottle (CGM) - Bệnh đốm lá xanh',
    'Cassava Mosaic Disease (CMD) - Bệnh khảm lá sắn',
    'Healthy - Cây khỏe mạnh'
]

# --- CHỨC NĂNG TẢI ẢNH ---
uploaded_files = st.file_uploader(
    "Kéo thả ảnh hoặc Chụp ảnh trực tiếp", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Ảnh chụp: {uploaded_file.name}", use_container_width=True)
        
        # Tiền xử lý (380x380 cho EfficientNet-B4)
        img_input = image.resize((380, 380))
        img_input = np.array(img_input)
        if img_input.shape[-1] == 4: img_input = img_input[..., :3]
        img_input = np.expand_dims(img_input / 255.0, axis=0)
        
        # Dự đoán
        with st.spinner('Đang phân tích dữ liệu...'):
            prediction = model.predict(img_input)
            idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
        # Hiển thị kết quả
        result_text = class_names[idx]
        
        if "Healthy" in result_text:
            st.success(f"✅ **Dự đoán:** {result_text}")
        else:
            st.error(f"⚠️ **Dự đoán:** {result_text}")
            
        st.write(f"📊 **Độ tin cậy:** {confidence:.2f}%")
        st.divider()

st.caption("Ứng dụng chạy trên nền tảng Streamlit Cloud - Dữ liệu thời gian thực.")