import torch
import torch.nn as nn
import timm
from models.timm_resnet import get_resnet_timm, get_resnet50_timm, get_resnet34_timm, get_resnet101_timm, \
    get_resnet_canonizer

# timm_vit
from models.timm_vit import get_vit_b_16, get_vit_s_16, get_convnext_base, get_vgg19, get_densenet, get_clip_vit_base

# dino
from models.model_dino import get_dinov3_vit_l_16

def get_resnet50_cub(n_class=None, ckpt_path=None):
    model = timm.create_model(
        'hf-hub:anonauthors/cub200-resnet50',
        pretrained=True
        )
    return model


MODELS = {
    "resnet50_cub": get_resnet50_cub,
    
    # resnet_timm
    "resnet50_timm": get_resnet50_timm,
    "resnet34_timm": get_resnet34_timm,
    "resnet101_timm": get_resnet101_timm,
    
    # vit_timm
    "vit_b_16_timm": get_vit_b_16,
    
    # others
    "vit_s_16_timm": get_vit_s_16,
    "convnext_timm": get_convnext_base,
    "vgg19_timm": get_vgg19,
    "clip_vit_b_16_timm": get_clip_vit_base,
    "densenet_timm": get_densenet,
    
    # dino
    "vit_l_16_dinov3": get_dinov3_vit_l_16,
}

CANONIZERS = {
    # resnet_timm
    "resnet50_timm": get_resnet_canonizer,
    "resnet34_timm": get_resnet_canonizer,
    "resnet101_timm": get_resnet_canonizer,
}


FEATURE_DIMS = {
    'resnet50': {
        'block_0': 256,
        'block_1': 512, 
        'block_2': 1024,
        'block_3': 2048,
    },
    'resnet18': {
        'block_0': 64,
        'block_1': 128,
        'block_2': 256,
        'block_3': 512,
    },
    'small_cnn': {
        'block_0': 16,
        'block_1': 32,
    },
    'resnet34_timm': {
        'block_0': 64,
        'block_1': 128,
        'block_2': 256,
        'block_3': 512,
        
        
        'layer1.0': 64,
        'layer1.1': 64,
        'layer1.2': 64,
        'layer2.0': 128,
        'layer2.1': 128,
        'layer2.2': 128,
        'layer2.3': 128,
        'layer3.0': 256,
        'layer3.1': 256,
        'layer3.2': 256,
        'layer3.3': 256,
        'layer3.4': 256,
        'layer3.5': 256,
        'layer4.0': 512,
        'layer4.1': 512,
        'layer4.2': 512,
    },
    
    'resnet50_timm': {
        'block_0': 256,
        'block_1': 512,
        'block_2': 1024,
        'block_3': 2048,

        'layer1': 256,
        'layer2': 512,
        'layer3': 1024,
        'layer4': 2048,

        'layer1.0': 256,
        'layer1.1': 256,
        'layer1.2': 256,
        'layer2.0': 512,
        'layer2.1': 512,
        'layer2.2': 512,
        'layer2.3': 512,
        'layer3.0': 1024,
        'layer3.1': 1024,
        'layer3.2': 1024,
        'layer3.3': 1024,
        'layer3.4': 1024,
        'layer3.5': 1024,
        'layer4.0': 2048,
        'layer4.1': 2048,
        'layer4.2': 2048,
    },

    'resnet101_timm': {
        'block_0': 256,
        'block_1': 512,
        'block_2': 1024,
        'block_3': 2048,
    },
    'resnet50_torchvision': {
        'layer1': 256,
        'layer2': 512, 
        'layer3': 1024,
        'layer4': 2048,
    },
    
    'resnet50_dino': {
        'layer1': 256,
        'layer2': 512, 
        'layer3': 1024,
        'layer4': 2048,
    },
    
    # efficientnet_timm
    'efficientnet_b0_timm': {
        'blocks.0': 16,
        'blocks.1': 24,
        'blocks.2': 40,
        'blocks.4': 112,
        'blocks.6': 320,
        
        'blocks.3': 80,
        'blocks.5': 192,
    },

    'efficientnet_b1_timm': {
        'blocks.0': 16,
        'blocks.1': 24,
        'blocks.2': 40,
        'blocks.3': 80,
        'blocks.4': 112,
        'blocks.5': 192,
        'blocks.6': 320,
    },

    'efficientnet_b2_timm': {
        'blocks.0': 16,
        'blocks.1': 24,
        'blocks.2': 48,
        'blocks.3': 88,
        'blocks.4': 120,
        'blocks.5': 208,
        'blocks.6': 352,
    },

    'efficientnet_b3_timm': {
        'blocks.0': 24,
        'blocks.1': 32,
        'blocks.2': 48,
        'blocks.3': 96,
        'blocks.4': 136,
        'blocks.5': 232,
        'blocks.6': 384,
    },

    # VGG_timm
    'vgg16_bn_timm': {
        'features.2': 64,
        'features.5': 64,
        'features.9': 128,
        'features.12': 128,
        'features.16': 256,
        'features.19': 256,
        'features.22': 256,
        'features.26': 512,
        'features.29': 512,
        'features.32': 512,
        'features.36': 512,
        'features.39': 512,
        'features.42': 512,
    },

    'vgg19_bn_timm': {
        'features.2': 64,
        'features.5': 64,
        'features.9': 128,
        'features.12': 128,
        'features.16': 256,
        'features.19': 256,
        'features.22': 256,
        'features.26': 512,
        'features.29': 512,
        'features.32': 512,
        'features.36': 512,
        'features.39': 512,
        'features.42': 512,
    },
    
    # ViT_b_32_torchvision
    'vit_b_32_torchvision': {
        'layers_0': 768,
        'layers_1': 768,
        'layers_2': 768,
        'layers_3': 768,
        'layers_4': 768,
        'layers_5': 768,
        'layers_6': 768,
        'layers_7': 768,
        'layers_8': 768,
        'layers_9': 768,
        'layers_10': 768,
        'layers_11': 768,
    },
    
    # ViT_b_16_timm
    'vit_b_16_timm': {
        'layers_0': 768,
        'layers_1': 768,
        'layers_2': 768,
        'layers_3': 768,
        'layers_4': 768,
        'layers_5': 768,
        'layers_6': 768,
        'layers_7': 768,
        'layers_8': 768,
        'layers_9': 768,
        'layers_10': 768,
        'layers_11': 768,

        'blocks.1': 768,
        'blocks.2': 768,
        'blocks.3': 768,
        'blocks.4': 768,
        'blocks.5': 768,
        'blocks.6': 768,
        'blocks.7': 768,
        'blocks.8': 768,
        'blocks.9': 768,
        'blocks.10': 768,
        'blocks.11': 768,
    },

    # vit_s_16_timm
    'vit_s_16_timm': {
        'blocks.0': 384,
        'blocks.1': 384,
        'blocks.2': 384,
        'blocks.3': 384,
        'blocks.4': 384,
        'blocks.5': 384,
        'blocks.6': 384,
        'blocks.7': 384,
        'blocks.8': 384,
        'blocks.9': 384,
        'blocks.10': 384,
        'blocks.11': 384,
    },

    # convnext
    'convnext_timm': {
        'stages.0': 128,
        'stages.1': 256,
        'stages.2': 512,
        'stages.3': 1024,


        'stages.0.blocks.0': 128,
        'stages.0.blocks.1': 128,
        'stages.0.blocks.2': 128, 
        'stages.1.blocks.0': 256,
        'stages.1.blocks.1': 256, 
        'stages.1.blocks.2': 256, 
        'stages.2.blocks.0': 512,
        'stages.2.blocks.1': 512,
        'stages.2.blocks.2': 512,
        'stages.2.blocks.3': 512,
        'stages.2.blocks.4': 512,
        'stages.2.blocks.5': 512,
        'stages.2.blocks.6': 512,
        'stages.2.blocks.7': 512,
        'stages.2.blocks.8': 512,
        'stages.2.blocks.9': 512,
        'stages.2.blocks.10': 512,
        'stages.2.blocks.11': 512,
        'stages.2.blocks.12': 512,
        'stages.2.blocks.13': 512,
        'stages.2.blocks.14': 512,
        'stages.2.blocks.15': 512,
        'stages.2.blocks.16': 512,
        'stages.2.blocks.17': 512,
        'stages.2.blocks.18': 512,
        'stages.2.blocks.19': 512,
        'stages.2.blocks.20': 512,
        'stages.2.blocks.21': 512,
        'stages.2.blocks.22': 512,
        'stages.2.blocks.23': 512,
        'stages.2.blocks.24': 512,
        'stages.2.blocks.25': 512,
        'stages.2.blocks.26': 512,
        'stages.3.blocks.0': 1024,
        'stages.3.blocks.1': 1024, 
        'stages.3.blocks.2': 1024, 
    },

    # vgg19
    'vgg19_timm': {
        # Correct - ReLU
        'features.1': 64,
        'features.3': 64,
        'features.6': 128,
        'features.8': 128,
        'features.11': 256,
        'features.13': 256,
        'features.15': 256,
        'features.17': 256,
        'features.20': 512,
        'features.22': 512,
        'features.24': 512,
        'features.26': 512,
        'features.29': 512,
        'features.31': 512,
        'features.33': 512,
        'features.35': 512,
        
        # Wrong
        # 'features.1': 64,
        # 'features.3': 64,
        # 'features.6': 128,
        # 'features.8': 128,
        # 'features.11': 256,
        # 'features.13': 256,
        # 'features.15': 256,
        # 'features.18': 256,
        # 'features.20': 512,
        # 'features.22': 512,
        # 'features.25': 512,
        # 'features.27': 512,
        # 'features.29': 512,
        # 'features.32': 512,
        # 'features.34': 512,
        # 'features.36': 512,
    },

    # clip_vit_b_16_timm
    'clip_vit_b_16_timm': {
        'blocks.1': 768,
        'blocks.2': 768,
        'blocks.3': 768,
        'blocks.4': 768,
        'blocks.5': 768,
        'blocks.6': 768,
        'blocks.7': 768,
        'blocks.8': 768,
        'blocks.9': 768,
        'blocks.10': 768,
        'blocks.11': 768,
    },
    
    "vit_l_16_dinov3": {
       # Interpolate all block indices from 0 to 23
       **{f"backbone.blocks.{i}": 1024 for i in range(24)},
    },

    'densenet_timm': {
        'features.denseblock2.denselayer1': 32,
        'features.denseblock3.denselayer12': 32,
        'features.denseblock4.denselayer16': 32,
    },
}


def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return [CANONIZERS[model_name]()]


def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")

# Create encoder to get intermediate layer outputs
class IntermediateLayerEncoder(nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.activation = {}
        self.hook = None
        
        # Register forward hook to get intermediate layer output
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Get the target layer and register hook
        for name, layer in self.model.named_modules():
            if name == layer_name:
                self.hook = layer.register_forward_hook(get_activation(layer_name))
                break
                
    def forward(self, x):
        _ = self.model(x)
        return self.activation[self.layer_name]
    
    def __del__(self):
        # Remove hook when object is destroyed
        if self.hook is not None:
            self.hook.remove()
