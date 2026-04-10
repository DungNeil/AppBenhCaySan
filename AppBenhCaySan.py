import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="AppBenhCaySan - Nhận Diện Bệnh Sắn",
    page_icon="🍃",
    layout="centered"
)

# Tắt bớt log thừa của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.title("🍃 AppBenhCaySan")
st.write("Hệ thống AI dự đoán bệnh lá sắn sẵn sàng.")

# --- TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    # Sử dụng bản 2.16+ thì để mặc định là tốt nhất
    model = tf.keras.models.load_model('best_model_cassava.h5', compile=False)
    return model

with st.spinner('Đang tải bộ não AI...'):
    model = load_model()

# --- DANH SÁCH NHÃN ---
# Lưu ý: Nếu kết quả vẫn sai tên bệnh, hãy thử đổi thứ tự các nhãn này
class_names = [
    'Cassava Bacterial Blight (CBB) - Bệnh cháy lá',
    'Cassava Brown Streak Disease (CBSD) - Bệnh sọc nâu thân',
    'Cassava Green Mottle (CGM) - Bệnh đốm lá xanh',
    'Cassava Mosaic Disease (CMD) - Bệnh khảm lá sắn',
    'Healthy - Cây khỏe mạnh'
]

# --- CHỨC NĂNG TẢI ẢNH ---
uploaded_files = st.file_uploader(
    "Kéo ảnh hoặc chụp ảnh trực tiếp", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Ảnh chụp: {uploaded_file.name}", use_container_width=True)
        
        # --- TIỀN XỬ LÝ CHUẨN ---
        with st.spinner('Đang phân tích dữ liệu...'):
            # 1. Resize về đúng kích thước model
            img_input = image.resize((380, 380))
            img_input = np.array(img_input)
            
            # 2. Xử lý kênh màu (Bỏ Alpha nếu có)
            if img_input.shape[-1] == 4: 
                img_input = img_input[..., :3]
            
            # 3. Chuẩn hóa (Thử dùng chế độ giữ nguyên 0-255 nếu EfficientNet)
            # Nếu kết quả vẫn sai, bạn thử thêm / 255.0 vào sau img_input
            img_input = np.expand_dims(img_input, axis=0)
            img_input = img_input.astype('float32') 
            
            # 4. Dự đoán
            prediction = model.predict(img_input)
            idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
        # --- HIỂN THỊ KẾT QUẢ ---
        result_text = class_names[idx]
        
        if "Healthy" in result_text:
            st.success(f"✅ **Dự đoán:** {result_text}")
        else:
            st.error(f"⚠️ **Dự đoán:** {result_text}")
            
        st.write(f"📊 **Độ tin cậy:** {confidence:.2f}%")
        st.divider()

st.caption("Ứng dụng khởi chạy Streamlit Cloud trên nền tảng - Thực tế dữ liệu thời gian thực.")