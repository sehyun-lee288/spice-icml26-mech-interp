
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

# ---------------------------------------------------------
# 1. Provided Custom Dataset Class
# ---------------------------------------------------------
class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                return False
        return True

    def _download(self):
        import tarfile
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.url, self.root, self.filename) #, self.tgz_md5)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # 1-based -> 0-based
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# ---------------------------------------------------------
# 2. DINO Model with Partial Unfreeze (Last Block + Head)
# ---------------------------------------------------------
class DinoTransferModel(nn.Module):
    def __init__(self, num_classes=200, model_name='dinov2_vitb14'):
        super().__init__()
        # Load Backbone (Facebook Research Hub)
        print(f"Loading {model_name}...")
        # self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.backbone = torch.hub.load('/project/dinov3', 
            'dinov3_vits16', source='local', weights='/project/dinov3/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')

        
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)

        # --- Freezing Logic ---
        # 1. Freeze All
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze Last Block (blocks[-1])
        print("Unfreezing the last transformer block...")
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
            
        # 3. Unfreeze Norm Layer (Final normalization)
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        # 4. Head is trainable by default

    def forward(self, x):
        # DINOv2 forward returns features. We use the CLS token strategy.
        # Note: Implementation detail might vary, usually it's x_norm_clstoken
        features = self.backbone(x)
        return self.head(features)

# ---------------------------------------------------------
# 3. Configuration & Training Setup
# ---------------------------------------------------------

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10  # 체크포인트 저장 및 뉴런 분석을 위해 넉넉히
LEARNING_RATE_BACKBONE = 1e-5 # 미세 조정용 낮은 LR
LEARNING_RATE_HEAD = 1e-4     # 신규 학습용 높은 LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (ImageNet Stats)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset & Loader
root_dir = '/project/CUB' # 데이터 저장 경로
train_dataset = Cub2011(root=root_dir, train=True, transform=train_transform, download=True)
val_dataset = Cub2011(root=root_dir, train=False, transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast # Mixed Precision

# ---------------------------------------------------------
# 1. Hyperparameters & Setup
# ---------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20       # CUB는 데이터가 적어 금방 수렴하거나 과적합되므로 20~30 정도면 충분
ACCUMULATION_STEPS = 1 # ViT-B는 보통 32 배치 가능하므로 1로 설정 (VRAM 부족시 늘리세요)

# Differential Learning Rate
LR_BACKBONE = 1e-5  # 이미 학습된 지식 보존 (미세 조정)
LR_HEAD = 1e-3      # 새로운 클래스 빠르게 학습

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler() # AMP Scaler

# ---------------------------------------------------------
# 2. Model & Optimizer Setup (CUB-200)
# ---------------------------------------------------------
# 앞서 정의한 클래스 사용 (DinoTransferModel, Cub2011)
model = DinoTransferModel(num_classes=200, model_name='dinov2_vitb14').to(DEVICE)

optimizer = optim.AdamW([
    {'params': model.backbone.blocks[-1].parameters(), 'lr': LR_BACKBONE},
    {'params': model.backbone.norm.parameters(),       'lr': LR_BACKBONE},
    {'params': model.head.parameters(),                'lr': LR_HEAD}
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# 3. Utility Functions
# ---------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Top-k accuracy 계산"""
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

def train_one_epoch(epoch, model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # AMP Forward
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / ACCUMULATION_STEPS

        # Backward
        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * ACCUMULATION_STEPS
    
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = len(loader)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            losses += loss.item()
            
            # Calculate Top-1 & Top-5
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1_acc += acc1.item()
            top5_acc += acc5.item()
            
    return losses / total_batches, top1_acc / total_batches, top5_acc / total_batches

# ---------------------------------------------------------
# 4. Main Execution Loop
# ---------------------------------------------------------
print(f"Start Training CUB-200 (Last Block Finetuning) on {DEVICE}...")

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
    
    # --- Validate ---
    val_loss, val_top1, val_top5 = validate(model, val_loader, criterion)
    
    print(f"[Epoch {epoch}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}%")
    
    # --- Save Best Model ---
    if val_top1 > best_acc:
        print(f"  --> New Record! (Old: {best_acc:.2f}% -> New: {val_top1:.2f}%)")
        best_acc = val_top1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, "dino_cub200_best.pth")
    
    # --- Save Checkpoint for Neuron Analysis ---
    # 뉴런 변화 추적을 위해 특정 epoch마다 저장
    if epoch % 1 == 0:
        torch.save(model.state_dict(), f"/project/results/dinov3/CUB/vit_s/dino_cub200_epoch_{epoch:03d}.pth")

print(f"\nTraining Finished. Final Best Top-1 Acc: {best_acc:.2f}%")