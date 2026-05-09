import torch
import torch.nn as nn

class DinoVitLTransfer(nn.Module):
    def __init__(self, num_classes=101, ckpt_path=None):
        super().__init__()
        print("Loading DINOv3 ViT-L (Large)...")
        self.backbone = torch.hub.load('/project/dinov3', 'dinov3_vitl16', source='local', 
                                       weights='/project/dinov3/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        if ckpt_path is not None:
            # Load checkpoint weights (ckpt_path) for head and last block's MLP/norm if available
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            # Remove possible 'module.' prefix if present (common for DataParallel checkpoints)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[len('module.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v

            # Attempt to load: backbone (last block and norm) + head
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded checkpoint from {ckpt_path}")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
            
            self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    
def get_dinov3_vit_l_16(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = DinoVitLTransfer(ckpt_path=ckpt_path)
    return model