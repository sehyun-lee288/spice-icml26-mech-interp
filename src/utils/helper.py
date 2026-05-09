import os

import torch
import yaml
from typing import List
from transformers import AutoProcessor


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
        config["wandb_id"] = os.path.basename(config_path)[:-5]
    return config


def get_layer_names_model(model: torch.nn.Module, model_name: str) -> List[str]:
    """
    Get layer names of a model.
    :param model:   model
    :param model_name:  model name (e.g. vgg16)
    :return:
    """
    if ("resnet" in model_name) and ("timm" in model_name):
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        layer_names = layer_names + ['layer1.0', 'layer1.1', 'layer1.2',
               'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
               'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
               'layer4.0', 'layer4.1', 'layer4.2',
               ]
    elif ("resnet" in model_name) and ("torchvision" in model_name):
        layer_names = ["layer1", "layer2", "layer3", "layer4"] # HARD CODING. TODO: support dynamic
        layer_names = layer_names + ['layer1.0', 'layer1.1', 'layer1.2', 
               'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
               'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
               'layer4.0', 'layer4.1', 'layer4.2', 
               ]
    elif ("resnet" in model_name):
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        layer_names = layer_names + ['layer1.0', 'layer1.1', 'layer1.2', 
               'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
               'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
               'layer4.0', 'layer4.1', 'layer4.2', 
        ]
    elif "dino" in model_name:
        layer_names = ["layer1", "layer2", "layer3", "layer4"] 
    elif "efficientnet" in model_name:
        layer_names = ["blocks.0", "blocks.1", "blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6"]
    elif "vgg19" in model_name:
        layer_names = [
            "features.1", "features.3", "features.6", "features.8",
            "features.11", "features.13", "features.15", "features.18", 
            "features.20", "features.22", "features.25", "features.29", 
            "features.32", "features.34", "features.36"]
    elif "convnext" in model_name:
        layer_names = [
                "stages.0", "stages.1", "stages.2", "stages.3"]
    elif "vit" in model_name:
        layer_names = [
            "blocks.0",
            "blocks.1",
            "blocks.2", 
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
            "blocks.7", 
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11"
        ]
    else:
        raise NotImplementedError
    return layer_names


class InspectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_pil_list, processor: AutoProcessor = None):
        self.data_pil_list=data_pil_list
        self.processor=processor
        self.length=len(data_pil_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        sample=self.data_pil_list[idx]
        if self.processor:
            sample=self.processor(images=self.data_pil_list[idx], return_tensors="pt")
            return sample['pixel_values'].squeeze()
        
        return sample.squeeze()
        
            