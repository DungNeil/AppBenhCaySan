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
# Chuyển layout sang 'wide' để đủ chỗ hiển thị 5 cột ngang
st.set_page_config(
    page_title="AI Chẩn Đoán Bệnh Sắn",
    page_icon="🌱",
    layout="wide" 
)

# CSS Custom cho các Thẻ kết quả (Cards)
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .healthy { background-color: #d4edda; color: #155724; border-left: 5px solid #28a745; }
    .disease { background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
    .disease-name-vn { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    .disease-name-en { font-size: 12px; font-style: italic; opacity: 0.8; }
    .conf-score { font-size: 14px; margin-top: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 AI Chẩn Đoán Bệnh Lá Sắn (Hệ thống xử lý hàng loạt)")
st.write("Tải lên một hoặc nhiều ảnh cùng lúc để hệ thống tự động phân tích và đưa ra phác đồ chẩn đoán.")
st.write("---")

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

# --- 2. TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b4()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"❌ Không tìm thấy file mô hình '{MODEL_PATH}'.")
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
tab1, tab2 = st.tabs(["📁 Tải hàng loạt ảnh", "📸 Chụp ảnh tại vườn"])

with tab1:
    # Bật tính năng accept_multiple_files
    uploaded_files = st.file_uploader("Kéo thả nhiều ảnh lá sắn vào đây...", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
with tab2:
    camera_upload = st.camera_input("Sử dụng Camera")

# Gom tất cả ảnh cần xử lý vào 1 list
images_to_process = []
if uploaded_files:
    images_to_process.extend(uploaded_files)
if camera_upload:
    images_to_process.append(camera_upload)

# --- 5. HỆ THỐNG XỬ LÝ LƯỚI 5 CỘT ---
if images_to_process:
    st.info(f"📁 Đã nhận dạng được **{len(images_to_process)}** bức ảnh chờ xử lý.")
    
    if st.button('🚀 KÍCH HOẠT HỆ THỐNG CHẨN ĐOÁN'):
        model, device = load_model()
        
        if model:
            st.success("✅ Hệ thống đang tiến hành quét...")
            
            # Thuật toán cắt danh sách ảnh thành từng nhóm 5 tấm
            chunk_size = 5
            for i in range(0, len(images_to_process), chunk_size):
                cols = st.columns(5) # Tạo 5 cột trên 1 hàng ngang
                chunk = images_to_process[i : i + chunk_size]
                
                # Rải từng ảnh vào từng cột tương ứng
                for j, file_data in enumerate(chunk):
                    with cols[j]:
                        try:
                            # 1. Hiển thị ảnh
                            image = Image.open(file_data)
                            st.image(image, use_container_width=True)
                            
                            # 2. Phân tích AI
                            with st.spinner('Đang quét...'):
                                input_tensor = preprocess_image(image).to(device)
                                with torch.no_grad():
                                    output = model(input_tensor)
                                    prob = torch.nn.functional.softmax(output, dim=1)
                                    confidence, predicted = torch.max(prob, 1)
                                    
                            res_idx = predicted.item()
                            conf_score = confidence.item()
                            
                            # 3. Tách chuỗi Tên bệnh để làm đẹp
                            full_name = CLASS_NAMES[res_idx]
                            name_en, name_vn = full_name.split(' - ')
                            
                            # 4. Hiển thị Thẻ kết quả (Màu Xanh nếu khỏe, Đỏ nếu bệnh)
                            status_class = "healthy" if res_idx == 4 else "disease"
                            
                            st.markdown(f"""
                            <div class="result-card {status_class}">
                                <div class="disease-name-vn">{name_vn}</div>
                                <div class="disease-name-en">{name_en}</div>
                                <div class="conf-score">Độ tự tin: {conf_score*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Thanh Progress Bar nhỏ ở dưới
                            st.progress(conf_score)
                            
                        except Exception as e:
                            st.error(f"Lỗi đọc ảnh: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2026 - Hệ thống AI hỗ trợ Nông nghiệp Thông minh</div>", unsafe_allow_html=True)