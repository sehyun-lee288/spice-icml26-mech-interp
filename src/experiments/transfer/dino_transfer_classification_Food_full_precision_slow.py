import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
# Scheduler 추가
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ---------------------------------------------------------
# 1. Configuration (Slow Down Settings)
# ---------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 10 # 천천히 오르므로 Epoch을 좀 더 늘림

# [중요] LR을 1/10 ~ 1/20 수준으로 대폭 낮춤
LEARNING_RATE_HEAD = 2e-5      # 기존 5e-4 -> 2e-5
LEARNING_RATE_BACKBONE = 1e-6  # 기존 1e-5 -> 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로그 및 체크포인트 경로
LOG_DIR = "./logs_slow"
CHECKPOINT_DIR = "/project/results/dinov3/Food/vit_l_slow"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "dino_slow_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "val_loss", "val_top1", "val_top5"])

# ---------------------------------------------------------
# 2. Data Loading (Same)
# ---------------------------------------------------------
# ... (기존 코드와 동일하므로 생략. Dataset, DataLoader 부분 유지) ...
# (실행 시에는 위 코드의 Dataset/Loader 부분을 그대로 가져오세요)
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
train_dataset = datasets.Food101(root='./data', split='train', transform=train_transform, download=True)
val_dataset = datasets.Food101(root='./data', split='test', transform=val_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ---------------------------------------------------------
# 3. Model Setup (Targeting MLP Only)
# ---------------------------------------------------------
class DinoVitLTransfer(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        print("Loading DINOv3 ViT-L (Large)...")
        self.backbone = torch.hub.load('/project/dinov3', 'dinov3_vitl16', source='local', 
                                       weights='/project/dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

        # --- [전략 3] MLP만 학습 (Attention은 Freeze) ---
        # 1. 전체 Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 2. Last Block의 'mlp'와 'norm'만 Unfreeze (attn은 끔)
        # 구조: blocks[-1].ls1, .attn, .ls2, .mlp
        last_block = self.backbone.blocks[-1]
        
        # LayerNorms
        for param in last_block.norm1.parameters(): param.requires_grad = True
        for param in last_block.norm2.parameters(): param.requires_grad = True
        
        # MLP (Feed Forward) - 여기가 개념 저장소라고 가정
        for param in last_block.mlp.parameters(): param.requires_grad = True
        
        # Final Norm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        # Head
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Accuracy 함수 (기존 동일)
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

# ---------------------------------------------------------
# 4. Optimizer & Scheduler (Warmup Added)
# ---------------------------------------------------------
# 파라미터 그룹 분리
mlp_params = [p for n, p in model.backbone.blocks[-1].mlp.named_parameters()]
norm_params = [p for n, p in model.backbone.named_parameters() if 'norm' in n and p.requires_grad]
head_params = [p for p in model.head.parameters()]

optimizer = optim.AdamW([
    {'params': mlp_params,  'lr': LEARNING_RATE_BACKBONE},
    {'params': norm_params, 'lr': LEARNING_RATE_BACKBONE},
    {'params': head_params, 'lr': LEARNING_RATE_HEAD}
], weight_decay=0.05) # Weight Decay를 좀 높여서 과적합 방지

criterion = nn.CrossEntropyLoss()

# [전략 1] Warmup Scheduler
# 처음 5 epoch 동안은 LR이 0에서 목표치까지 서서히 증가 (Slow Start)
scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5) 

# ---------------------------------------------------------
# 5. Validation Function
# ---------------------------------------------------------
def validate(model, val_loader, criterion):
    model.eval()
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = len(val_loader)
    
    with torch.no_grad():
        # Validation은 빠르니까 Tqdm 생략 혹은 간단히
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses += loss.item()
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1_acc += acc1.item()
            top5_acc += acc5.item()

    return losses / total_batches, top1_acc / total_batches, top5_acc / total_batches

# ---------------------------------------------------------
# 6. Training Loop with Intra-Epoch Checkpointing
# ---------------------------------------------------------
global_step = 0
SAVE_INTERVAL_STEPS = 50 # [전략 2] 100 스텝마다 저장 (Epoch 내 변화 관찰용)

print(f"Start Slow Training (Target: MLP Only, Warmup, Low LR)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100) as t:
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            t.set_postfix(loss=running_loss/(batch_idx+1), lr=optimizer.param_groups[-1]['lr'])

            # --- [전략 2] Intra-Epoch Checkpointing ---
            # 초반(Epoch 1~2)에는 자주 저장해서 뉴런 변화를 세밀하게 포착
            if epoch <= 2 and (batch_idx + 1) % SAVE_INTERVAL_STEPS == 0:
                # Validation은 시간이 걸리므로 Step 단위 저장시에는 생략하거나 약식으로 진행
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                
                # 로그 기록 (Step 단위)
                with open(LOG_FILE, mode='a', newline='') as f:
                    csv.writer(f).writerow([global_step, epoch, loss.item(), "", "", ""])

    # Epoch 끝날 때마다 Scheduler Step
    scheduler.step()
    
    # Epoch 단위 Full Validation
    val_loss, val_top1, val_top5 = validate(model, val_loader, criterion)
    
    print(f"\n[Epoch {epoch}] Val Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}% | LR: {optimizer.param_groups[-1]['lr']:.2e}")

    # Epoch 저장
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pth"))
    
    # 로그 기록 (Epoch 단위)
    with open(LOG_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([global_step, epoch, running_loss/len(train_loader), val_loss, val_top1, val_top5])

print("Finished.")