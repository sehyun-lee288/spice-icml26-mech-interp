import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE_HEAD = 5e-4
LEARNING_RATE_BACKBONE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [추가] 저장 주기 설정 (1 = 매 step 저장, 10 = 10 step 마다 저장)
# 주의: ViT-L 체크포인트는 큽니다. 매 스텝 저장은 디스크 용량을 폭발시킬 수 있습니다.
SAVE_INTERVAL = 100

# 경로 설정
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_EPOCH = os.path.join(LOG_DIR, "dino_vitl_food101_epoch_log.csv")
LOG_FILE_STEP = os.path.join(LOG_DIR, "dino_vitl_food101_step_log.csv") # [추가] 스텝별 로그

# 스텝별 체크포인트 저장 경로
CHECKPOINT_STEP_DIR = "/project/results/dinov3/Food/vit_l_fp32_steps"
os.makedirs(CHECKPOINT_STEP_DIR, exist_ok=True)

# 로그 파일 헤더 초기화
if not os.path.exists(LOG_FILE_EPOCH):
    with open(LOG_FILE_EPOCH, mode='w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_top1_acc", "val_top5_acc"])

if not os.path.exists(LOG_FILE_STEP):
    with open(LOG_FILE_STEP, mode='w', newline='') as f:
        csv.writer(f).writerow(["global_step", "epoch", "step_in_epoch", "loss", "train_top1", "train_top5"])

# ---------------------------------------------------------
# 2. Data Loading (유지)
# ---------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Downloading Food-101...")
train_dataset = datasets.Food101(root='./data', split='train', transform=train_transform, download=True)
val_dataset = datasets.Food101(root='./data', split='test', transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# 3. Model & Utility (유지)
# ---------------------------------------------------------
class DinoVitLTransfer(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        print("Loading DINOv3 ViT-L (Large)...")
        self.backbone = torch.hub.load('/project/dinov3',
            'dinov3_vitl16', source='local', weights='/project/dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

        for param in self.backbone.parameters(): param.requires_grad = False
        for param in self.backbone.blocks[-1].parameters(): param.requires_grad = True
        for param in self.backbone.norm.parameters(): param.requires_grad = True
        for param in self.head.parameters(): param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def accuracy(output, target, topk=(1,)):
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

optimizer = optim.AdamW([
    {'params': model.backbone.blocks[-1].parameters(), 'lr': LEARNING_RATE_BACKBONE},
    {'params': model.backbone.norm.parameters(),       'lr': LEARNING_RATE_BACKBONE},
    {'params': model.head.parameters(),                'lr': LEARNING_RATE_HEAD}
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# 4. Training Loop (Step-wise Logging & Saving 추가)
# ---------------------------------------------------------
def train(epoch, global_step):
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # CSV 파일 핸들 열기 (append mode)
    f_step_log = open(LOG_FILE_STEP, mode='a', newline='')
    writer_step = csv.writer(f_step_log)

    with tqdm(enumerate(train_loader), total=num_batches, desc=f"Train Epoch {epoch}", ncols=100) as t:
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            # [추가] 현재 배치의 Accuracy 계산
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # [추가] Step별 로그 기록
            writer_step.writerow([
                global_step, 
                epoch, 
                batch_idx + 1, 
                round(loss.item(), 6), 
                round(acc1.item(), 4), 
                round(acc5.item(), 4)
            ])
            
            # [추가] Step별 모델 저장 (SAVE_INTERVAL 마다)
            if global_step % SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_STEP_DIR, f"ckpt_step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                # (옵션) 너무 많은 파일 생성을 막기 위해 화면 출력은 생략하거나 간단히 표시
                # t.write(f"Saved checkpoint at step {global_step}")

            if batch_idx % 20 == 0:
                t.set_postfix(loss=loss.item(), acc=acc1.item())
    
    f_step_log.close()
    avg_loss = running_loss / num_batches
    return avg_loss, global_step

# ---------------------------------------------------------
# 5. Validation Function (유지)
# ---------------------------------------------------------
def validate(model, val_loader, criterion):
    model.eval() 
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = len(val_loader)
    
    with torch.no_grad():
        with tqdm(val_loader, total=total_batches, desc="Validating", ncols=100) as t:
            for inputs, targets in t:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses += loss.item()
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                t.set_postfix(loss=losses/(t.n+1), top1=top1_acc/(t.n+1))

    return losses / total_batches, top1_acc / total_batches, top5_acc / total_batches

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
best_acc = 0.0
global_step = 0 # 전체 스텝 카운트

print(f"Start Training (Save Interval: {SAVE_INTERVAL} steps)...")

for epoch in range(1, EPOCHS + 1):
    # 1. Train (global_step 전달 및 갱신)
    train_loss, global_step = train(epoch, global_step)
    
    # 2. Validate
    val_loss, val_top1, val_top5 = validate(model, val_loader, criterion)

    print(f"Epoch {epoch} Result: Train Loss {train_loss:.4f} | Val Acc {val_top1:.2f}%")
    
    # 3. Save Best Model (Epoch 단위)
    if val_top1 > best_acc:
        best_acc = val_top1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, "dino_vitl_food101_best_fp32.pth")
    
    # Epoch Log
    with open(LOG_FILE_EPOCH, mode='a', newline='') as f:
        csv.writer(f).writerow([epoch, train_loss, val_loss, val_top1, val_top5])

print(f"Finished. Best Acc: {best_acc:.2f}%")