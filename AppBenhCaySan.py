import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="AI Chẩn Đoán Bệnh Sắn",
    page_icon="🌱",
    layout="centered"
)

# Thêm CSS để giao diện trông hiện đại hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 AI Chẩn Đoán Bệnh Lá Sắn")
st.write("Giải pháp hỗ trợ nông dân phát hiện bệnh sớm bằng Trí tuệ nhân tạo.")

# --- 1. ĐỊNH NGHĨA THÔNG SỐ ---
CLASS_NAMES = [
    'Cassava Bacterial Blight (CBB) - Bệnh cháy lá do vi khuẩn',
    'Cassava Brown Streak Disease (CBSD) - Bệnh sọc nâu thân sắn',
    'Cassava Green Mottle (CGM) - Bệnh đốm lá xanh',
    'Cassava Mosaic Disease (CMD) - Bệnh khảm lá sắn',
    'Healthy - Cây khỏe mạnh'
]

MODEL_PATH = 'best_model_torchvision.pth'
IMG_SIZE = 380

# --- 2. TẢI MÔ HÌNH (CACHE ĐỂ CHẠY NHANH) ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Khởi tạo EfficientNet-B4
    model = models.efficientnet_b4()
    # Sửa lại lớp phân loại (5 lớp)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    
    # Kiểm tra file model có tồn tại không
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"❌ Không tìm thấy file {MODEL_PATH}! Hãy đảm bảo file này nằm cùng thư mục với app.py.")
        return None, None

# --- 3. TIỀN XỬ LÝ ẢNH ---
def preprocess_image(image):
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_np = np.array(image.convert('RGB'))
    return transform(image=image_np)['image'].unsqueeze(0)

# --- 4. GIAO DIỆN CHỌN ẢNH ---
# Cho phép chọn giữa tải file hoặc dùng camera (rất tiện khi ra đồng)
tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp ảnh trực tiếp"])

with tab1:
    file_upload = st.file_uploader("Chọn ảnh lá sắn từ máy...", type=['jpg', 'png', 'jpeg'])
with tab2:
    camera_upload = st.camera_input("Chụp ảnh lá sắn ngay bây giờ")

input_file = file_upload if file_upload is not None else camera_upload

if input_file is not None:
    image = Image.open(input_file)
    st.image(image, caption='Ảnh đã chọn', use_container_width=True)
    
    if st.button('🚀 Bắt đầu chẩn đoán'):
        model, device = load_model()
        
        if model:
            with st.spinner('Đang phân tích vết bệnh...'):
                input_tensor = preprocess_image(image).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(prob, 1)
                
                # --- 5. HIỂN THỊ KẾT QUẢ ---
                res_idx = predicted.item()
                conf_score = confidence.item()
                
                st.success("✅ Phân tích hoàn tất!")
                
                # Hiển thị kết quả nổi bật
                st.markdown(f"### Kết quả: **{CLASS_NAMES[res_idx]}**")
                st.markdown(f"### Độ tin cậy: **{conf_score*100:.2f}%**")
                
                # Thanh Progress bar thể hiện độ tin cậy
                st.progress(conf_score)
                
                # Đưa ra lời khuyên dựa trên kết quả
                if res_idx == 4: # Healthy
                    st.balloons()
                    st.info("Cây đang phát triển rất tốt. Hãy tiếp tục theo dõi định kỳ nhé!")
                else:
                    st.warning("Cảnh báo: Phát hiện triệu chứng bệnh. Bạn nên tham khảo ý kiến chuyên gia nông nghiệp để có biện pháp xử lý kịp thời.")

# --- 6. THÔNG TIN BỔ SUNG ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Cassava_leaf.jpg", caption="Lá sắn khỏe mạnh")
    st.info("""
    **Hướng dẫn chụp ảnh:**
    1. Chụp gần mặt lá (khoảng 20-30cm).
    2. Đảm bảo đủ ánh sáng.
    3. Đặt vết bệnh ở trung tâm ảnh.
    """)
    st.write("---")
    st.write("© 2026 - Đồ án tốt nghiệp: Nhận dạng bệnh trên lá cây")