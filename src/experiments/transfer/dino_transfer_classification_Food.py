import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast # Mixed Precision 필수
import csv

# tqdm 추가
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Configuration for ViT-L (Large)
# ---------------------------------------------------------
# ViT-L은 VRAM을 많이 차지하므로 배치 사이즈를 줄이고 Gradient Accumulation을 씁니다.
# 예: 배치 16 * 4번 누적 = 실제 배치 64 효과
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE_HEAD = 5e-4      # Head는 적당히 빠르게
LEARNING_RATE_BACKBONE = 1e-5  # Backbone은 아주 미세하게
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로그 디렉토리/파일 지정
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "dino_vitl_food101_log.csv")

# 로그 파일 헤더 초기화 (존재하지 않을 때만)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss",
            "val_loss", "val_top1_acc", "val_top5_acc"
        ])

# ---------------------------------------------------------
# 2. Data Loading (Food-101)
# ---------------------------------------------------------
# Food-101은 torchvision에 내장되어 있습니다.
# 이미지가 꽤 크고 다양하므로 224x224 (또는 가능하면 518) 리사이즈
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # 음식은 조명 영향 큼
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Downloading Food-101 (approx 5GB)... this may take a while.")
# download=True 하면 자동으로 받고 압축 풉니다.
train_dataset = datasets.Food101(root='./data', split='train', transform=train_transform, download=True)
val_dataset = datasets.Food101(root='./data', split='test', transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# 3. Dino ViT-L Model Setup
# ---------------------------------------------------------
class DinoVitLTransfer(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        # ViT-L (Large) 로드: 'dinov2_vitl14'
        print("Loading DINOv3 ViT-L (Large)...")
        self.backbone = torch.hub.load('/project/dinov3',
            'dinov3_vitl16', source='local', weights='/project/dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

        # ViT-L의 Embed Dim은 1024입니다. (Base는 768)
        self.embed_dim = self.backbone.embed_dim

        self.head = nn.Linear(self.embed_dim, num_classes)

        # --- Partial Freezing Strategy ---
        # 1. 전체 Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Last Block Unfreeze (ViT-L은 깊으므로 24번째 블록)
        # Food-101은 데이터가 많아서 마지막 2개 블록을 풀어도 됩니다. (여기선 1개만 예시)
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

        # 3. Norm Layer Unfreeze
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # 4. Head Unfreeze
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

model = DinoVitLTransfer(num_classes=101).to(DEVICE)

# ---------------------------------------------------------
# 4. Training Loop with Mixed Precision (AMP)
# ---------------------------------------------------------
optimizer = optim.AdamW([
    {'params': model.backbone.blocks[-1].parameters(), 'lr': LEARNING_RATE_BACKBONE},
    {'params': model.backbone.norm.parameters(),       'lr': LEARNING_RATE_BACKBONE},
    {'params': model.head.parameters(),                'lr': LEARNING_RATE_HEAD}
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss()
scaler = GradScaler('cuda') # AMP Scaler

def train(epoch):
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    with tqdm(enumerate(train_loader), total=num_batches, desc=f"Train Epoch {epoch}", ncols=100) as t:
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / ACCUMULATION_STEPS # 그라디언트 누적을 위한 나누기

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUMULATION_STEPS
                
            if batch_idx % 100 == 0:
                t.set_postfix(loss=running_loss/(batch_idx+1))
        
    avg_loss = running_loss / num_batches
    return avg_loss


# ---------------------------------------------------------
# Validation Function
# ---------------------------------------------------------
def validate(model, val_loader, criterion):
    model.eval() # 평가 모드 전환 (Dropout, BatchNorm 등 고정)

    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = len(val_loader)
    
    with torch.no_grad(): # Gradient 계산 비활성화 (메모리 절약)
        with tqdm(val_loader, total=total_batches, desc="Validating", ncols=100) as t:
            for inputs, targets in t:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                losses += loss.item()

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                # tqdm에는 batch별 평균 loss 표시
                t.set_postfix(loss=losses/(t.n+1), top1=top1_acc/(t.n+1), top5=top5_acc/(t.n+1))

    avg_loss = losses / total_batches
    avg_top1 = top1_acc / total_batches
    avg_top5 = top5_acc / total_batches

    return avg_loss, avg_top1, avg_top5

# ---------------------------------------------------------
# Main Training Loop with Validation & Logging
# ---------------------------------------------------------
best_acc = 0.0

print(f"Start Training ViT-L on Food-101 for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    # 1. Train
    train_loss = train(epoch) # 뒷부분에서 사용/로그를 위해 평균 loss 반환
    
    # 2. Validate
    val_loss, val_top1, val_top5 = validate(model, val_loader, criterion)

    print(f"Epoch {epoch} Result:")
    print(f"  - Train Loss: {train_loss:.4f}")
    print(f"  - Val Loss:  {val_loss:.4f}")
    print(f"  - Top-1 Acc: {val_top1:.2f}%")
    print(f"  - Top-5 Acc: {val_top5:.2f}%")
    
    # 3. Save Best Model (Checkpointing)
    # 연구용이므로 Top-1 Accuracy 기준으로 Best 모델 저장
    if val_top1 > best_acc:
        best_acc = val_top1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, "dino_vitl_food101_best.pth")
        print(f"  --> New Best Model Saved! ({best_acc:.2f}%)")
    
    # 뉴런 분석용 정기 체크포인트 (예: 매 Epoch 저장)
    save_dir = "/project/results/dinov3/Food/vit_l"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"dino_vitl_food101_epoch_{epoch}.pth"))

    # 4. Loss & Accuracy Logging (CSV)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            round(train_loss, 6),
            round(val_loss, 6),
            round(val_top1, 6),
            round(val_top5, 6),
        ])

print(f"Training Finished. Best Top-1 Accuracy: {best_acc:.2f}%")