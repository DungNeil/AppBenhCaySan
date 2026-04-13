import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models # Dùng thư viện chính chủ của PyTorch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG VÀ SIÊU THAM SỐ
# ==========================================
DATA_DIR = '/kaggle/input/competitions/cassava-leaf-disease-classification/'
TRAIN_CSV = DATA_DIR + 'train.csv'
TRAIN_IMG_DIR = DATA_DIR + 'train_images/'
WORKING_DIR = '/kaggle/working/' 

IMG_SIZE = 380
BATCH_SIZE = 16  # Đã hạ xuống 16 để chống lỗi Out of Memory
NUM_CLASSES = 5
EPOCHS = 10 
SEED = 42

# Tên các loại bệnh (Song ngữ - Phục vụ làm App và Báo cáo Đồ án)
CLASS_NAMES = [
    'Cassava Bacterial Blight (CBB) - Bệnh cháy lá do vi khuẩn',
    'Cassava Brown Streak Disease (CBSD) - Bệnh sọc nâu thân sắn',
    'Cassava Green Mottle (CGM) - Bệnh đốm lá xanh',
    'Cassava Mosaic Disease (CMD) - Bệnh khảm lá sắn',
    'Healthy - Cây khỏe mạnh'
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(SEED)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & AUGMENTATION
# ==========================================
df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)

train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: 
            image = self.transform(image=image)['image']
        return image, torch.tensor(row['label'], dtype=torch.long)

train_loader = DataLoader(CassavaDataset(train_df, TRAIN_IMG_DIR, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(CassavaDataset(val_df, TRAIN_IMG_DIR, val_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. MÔ HÌNH (TORCHVISION) VÀ HÀM MẤT MÁT
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo EfficientNet-B4 bằng torchvision cực kỳ ổn định
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

# Tùy chỉnh lớp phân loại cuối cùng cho bài toán 5 nhãn
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
print("🚀 BẮT ĐẦU HUẤN LUYỆN TRÊN:", device)
print("Sử dụng Torchvision Models - Đảm bảo ổn định!")
best_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_preds, best_labels = [], []

for epoch in range(EPOCHS):
    # Trạng thái Training
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = 100 * train_correct / train_total
    
    # Trạng thái Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], [] 
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * val_correct / val_total
    
    # Lưu lịch sử
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}%")
    
    # Lưu mô hình tốt nhất
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), WORKING_DIR + 'best_model_torchvision.pth')
        best_preds, best_labels = all_preds, all_labels 
    
    scheduler.step()

# ==========================================
# 5. XUẤT KẾT QUẢ CHO BÁO CÁO ĐỒ ÁN
# ==========================================
print("\n" + "="*70)
print("📊 BÁO CÁO KẾT QUẢ PHÂN LẠI (CLASSIFICATION REPORT)")
print("="*70)
print(classification_report(best_labels, best_preds, target_names=CLASS_NAMES))

# 5.1 Vẽ biểu đồ Loss & Accuracy
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='o')
plt.title('Biểu đồ Loss qua các Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
plt.title('Biểu đồ Accuracy qua các Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(WORKING_DIR + 'training_history.png', dpi=300)
plt.show()

# 5.2 Vẽ Ma trận nhầm lẫn (Confusion Matrix) chống đè chữ
plt.figure(figsize=(12, 10))
cm = confusion_matrix(best_labels, best_preds)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 12}) # Tăng cỡ chữ số bên trong ô

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=16, pad=20)
plt.ylabel('Nhãn Thực Tế (True Label)', fontweight='bold', fontsize=12)
plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(WORKING_DIR + 'confusion_matrix.png', dpi=300)
plt.show()

print("✅ Đã lưu file mô hình (best_model_torchvision.pth) và các biểu đồ vào thư mục Output của Kaggle.")
