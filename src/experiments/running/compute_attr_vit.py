import pickle
import os
import glob
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from safetensors.torch import load_file, save_file
from timm.data import create_transform, resolve_data_config
import torch
from torchmetrics import Accuracy

from dsets import get_imagenet
from models import get_fn_model_loader
from experiments.disentangling.attribution import AttributionExtractor

# Load model and dataset
model = get_fn_model_loader('vit_b_16_timm')(pretrained=True)
model.eval().cuda();

transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
transform = transform.transforms

dataset = get_imagenet(
    data_path='/project/data/external/ILSVRC/Data/CLS-LOC',
    preprocessing=True,
    split='val',
    transform=transform)

# Load highly activated sample indices
top_indices_path = '/project/results/top_activations/imagenet/vit_b_16_timm/top10pct/top_activations_blocks_11_output.safetensors'
top_indices = load_file(top_indices_path)['blocks.11_top_indices'] # (channel OR token_dim, samples)

# Set (input * gradient) extractor
device = 'cuda'
extractor = AttributionExtractor(model, device)
tgt_layer_module = model.blocks[11]
src_layer_module = model.blocks[10]

save_dir = '/project/results/attributions/imagenet/vit_b_16_timm/blocks.11'
os.makedirs(save_dir, exist_ok=True)

for target_neurons_slice in trange(768):
    subset = torch.utils.data.Subset(dataset, top_indices[target_neurons_slice])
    dataloader = torch.utils.data.DataLoader(subset, batch_size=256, 
                                         shuffle=False, num_workers=8, pin_memory=True)

    all_attributions = []
    
    for image, _ in dataloader:
        input_batch = image.cuda()

        attribution_map = extractor.compute_input_gradient(
                input_tensor=input_batch,
                tgt_layer=tgt_layer_module,
                src_layer=src_layer_module,
                tgt_neurons=target_neurons_slice)
        
        all_attributions.append(attribution_map.cpu())
    
    all_attributions = torch.cat(all_attributions, dim=0)    
    
    # Save attribution_map for all subset
    save_path = os.path.join(save_dir, f'attribution_{target_neurons_slice:03d}.safetensors')
    data_to_save = {'attribution': all_attributions}
    save_file(data_to_save, save_path)
