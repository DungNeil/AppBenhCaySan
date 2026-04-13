import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
import pandas as pd

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="AI Chẩn Đoán Bệnh Sắn",
    page_icon="🌱",
    layout="wide"
)

# --- QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
# Tạo "chìa khóa" để reset công cụ tải ảnh
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

# CSS Custom cho thẻ kết quả
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #1b5e20; color: white; font-weight: bold;
    }
    .file-name {
        font-size: 11px; color: #555; background: #eee;
        padding: 4px 8px; border-radius: 4px; margin-bottom: 8px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        text-align: center; font-weight: bold;
    }
    .result-card {
        padding: 12px; border-radius: 8px; margin-bottom: 20px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .healthy { background-color: #e8f5e9; color: #2e7d32; border-bottom: 4px solid #28a745; }
    .disease { background-color: #ffebee; color: #c62828; border-bottom: 4px solid #dc3545; }
    .nodata { background-color: #fff3cd; color: #856404; border-bottom: 4px solid #ffc107; } 
    
    .vn-name { font-size: 15px; font-weight: bold; margin-bottom: 2px;}
    .en-name { font-size: 12px; font-style: italic; margin-bottom: 8px; opacity: 0.8;}
    .conf-score { font-size: 13px; opacity: 0.9; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Dashboard Chẩn Đoán Bệnh Lá Sắn")
st.write("Hệ thống Trí tuệ Nhân tạo hỗ trợ phân tích và nhận diện bệnh tự động.")
st.warning("⚠️ **Lưu ý:** Hệ thống phân loại đóng (Closed-world Classification) được huấn luyện chuyên biệt trên hình ảnh Lá Sắn. Việc tải lên các hình ảnh không liên quan (đồ vật, màn hình máy tính, con người...) có thể dẫn đến hiện tượng 'Độ tự tin ảo' (Overconfidence) của hàm Softmax.")

# --- 1. THÔNG SỐ VÀ DANH SÁCH BỆNH ---
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
    return None, None

def preprocess(image):
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform(image=np.array(image.convert('RGB')))['image'].unsqueeze(0)

# --- 3. GIAO DIỆN NHẬP DỮ LIỆU ---
with st.sidebar:
    st.header("⚙️ Nguồn dữ liệu")
    tab1, tab2 = st.tabs(["📁 Tải ảnh", "📸 Camera"])
    
    with tab1:
        # Gắn chìa khóa vào hộp tải ảnh
        uploaded_files = st.file_uploader("Kéo thả ảnh vào đây...", type=['jpg','png','jpeg'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    with tab2:
        # Gắn chìa khóa vào camera
        camera_upload = st.camera_input("Chụp ảnh tại vườn", key=f"camera_{st.session_state.camera_key}")
        
    st.markdown("---")
    if st.button("🗑️ Xóa dữ liệu & Làm mới"):
        st.cache_resource.clear()
        # Đổi chìa khóa mới để ép Streamlit xóa sạch ảnh cũ trên giao diện
        st.session_state.uploader_key += 1
        st.session_state.camera_key += 1
        st.rerun()

images_to_process = []
if uploaded_files:
    images_to_process.extend(uploaded_files)
if camera_upload:
    images_to_process.append(camera_upload)

# --- 4. XỬ LÝ CHÍNH ---
if images_to_process:
    model, device = load_model()
    if model:
        results_list = []
        
        st.info(f"Đang chờ xử lý **{len(images_to_process)}** mẫu ảnh.")
        if st.button('🚀 BẮT ĐẦU PHÂN TÍCH'):
            
            with st.spinner('AI đang quét và phân tích dữ liệu...'):
                for file in images_to_process:
                    try:
                        img = Image.open(file)
                        input_tensor = preprocess(img).to(device)
                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = torch.nn.functional.softmax(output, dim=1)
                            conf, pred = torch.max(prob, 1)
                        
                        conf_score = conf.item()
                        
                        # --- BỘ LỌC NO DATA ---
                        CONF_THRESHOLD = 0.50  # Ngưỡng an toàn chung 50%
                        
                        if conf_score < CONF_THRESHOLD:
                            vn_name = "Không đủ dữ kiện"
                            en_name = "Low Confidence / OOD"
                            css_class = "nodata"
                            pred_id = -1 
                            conf_html = f'<div class="conf-score" style="color: #d32f2f;">Đạt: {round(conf_score*100, 2)}% (Cần > 50%)</div>'
                        else:
                            full_name = CLASS_NAMES[pred.item()]
                            en_name, vn_name = full_name.split(' - ')
                            css_class = "healthy" if pred.item() == 4 else "disease"
                            pred_id = pred.item()
                            conf_html = f'<div class="conf-score">Tự tin: {round(conf_score*100, 2)}%</div>'
                        
                        file_name_display = file.name if hasattr(file, 'name') and "camera_input" not in file.name else "Ảnh từ Camera"
                        
                        results_list.append({
                            "Tên file": file_name_display,
                            "Chẩn đoán": vn_name,
                            "Tiếng Anh": en_name,
                            "Mã bệnh": pred_id,
                            "conf_html": conf_html,
                            "img_data": img,
                            "css_class": css_class
                        })
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file ảnh: {e}")

            # --- 5. BẢNG THỐNG KÊ ---
            if len(results_list) > 0:
                df = pd.DataFrame(results_list)
                valid_df = df[df['Mã bệnh'] != -1]
                
                st.subheader("📈 Thống kê tổng quan")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số ảnh đã tải", len(df))
                
                if len(valid_df) > 0:
                    health_rate = len(valid_df[valid_df['Mã bệnh']==4]) / len(valid_df) * 100
                    m2.metric("Tỷ lệ cây khỏe (Hợp lệ)", f"{health_rate:.1f}%")
                    most_common = valid_df['Chẩn đoán'].mode()[0]
                    m3.metric("Loại phổ biến nhất", most_common)
                else:
                    m2.metric("Tỷ lệ cây khỏe", "N/A")
                    m3.metric("Loại phổ biến nhất", "Không có dữ liệu hợp lệ")

                st.divider()

                # --- 6. HIỂN THỊ LƯỚI ẢNH 5 CỘT ---
                st.subheader("🖼️ Chi tiết từng mẫu ảnh")
                chunk_size = 5
                for i in range(0, len(results_list), chunk_size):
                    cols = st.columns(5)
                    chunk = results_list[i : i + chunk_size]
                    
                    for j, res in enumerate(chunk):
                        with cols[j]:
                            st.markdown(f'<div class="file-name" title="{res["Tên file"]}">{res["Tên file"]}</div>', unsafe_allow_html=True)
                            st.image(res['img_data'], use_container_width=True)
                            
                            st.markdown(f"""
                                <div class="result-card {res['css_class']}">
                                    <div class="vn-name">{res['Chẩn đoán']}</div>
                                    <div class="en-name">{res['Tiếng Anh']}</div>
                                    {res['conf_html']}
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Chưa có dữ liệu để phân tích. Hãy tải ảnh lên và bấm nút.")

else:
    st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: white; border-radius: 10px; margin-top: 20px;'>
            <h3 style='color: #888;'>👈 Vui lòng thêm ảnh từ thanh công cụ bên trái để bắt đầu</h3>
        </div>
    """, unsafe_allow_html=True)