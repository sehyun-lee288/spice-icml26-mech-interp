import timm
import torch
import torch.hub

def get_vit_b_16(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('vit_base_patch16_224', 
                              pretrained='timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 
                              num_classes=n_class)
    return model

def get_vit_s_16(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('vit_small_patch16_224',
                              pretrained='timm/vit_small_patch16_224.augreg_in21k_ft_in1k',
                              num_classes=n_class)
    return model

def get_vgg19(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('vgg19',
                              pretrained='timm/vgg19.tv_in1k',
                              num_classes=n_class)
    return model

def get_convnext_base(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('convnext_base',
                              pretrained='timm/b_in22k_ft_in1k',
                              num_classes=n_class)
    return model

def get_densenet(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('timm/densenet121.tv_in1k',
                              pretrained='timm/densenet121.tv_in1k',
                              num_classes=n_class)
    return model

def get_clip_vit_base(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model('timm/vit_base_patch16_clip_224.openai',
                              pretrained='timm/vit_base_patch16_clip_224.openai',
                              num_classes=n_class)
    return model