import pickle
import os
import glob
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import load_file
from timm.data import create_transform, resolve_data_config
import torch
from torchmetrics import Accuracy

from dsets import get_imagenet
from models import get_fn_model_loader

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

dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, 
                                         shuffle=False, num_workers=8, pin_memory=True)


score_path = '/project/results/clustering/kmeans/blocks.11/score.pkl'
with open(score_path, 'rb') as f:
    scores = pickle.load(f)


# # Original Evaluation
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    targets = []
    preds = []
    for images, labels in tqdm(dataloader, total=len(dataloader)):
        out = model(images.cuda())
        pred = out.argmax(dim=1)
        
        targets.append(labels)
        preds.append(pred.detach().cpu())
        
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return targets, preds



targets, preds = evaluate(model, dataloader)

accuracy = Accuracy(task="multiclass", num_classes=1000)
acc = accuracy(preds, targets)



path = 'orig_pred.pkl'
with open(path, 'wb') as f:
    pickle.dump({
                     'targets': targets,
                     'preds': preds,
                     'acc': acc
                 }, f)


@torch.no_grad()
def evaluate(model, dataloader, neurons_to_ablate=None):
    if neurons_to_ablate is not None:
        def ablate_hook(module, input, output):
            output[:, :, neurons_to_ablate] = 0
            return output
        model.blocks[11].register_forward_hook(ablate_hook)
    
    model.eval()
    targets = []
    preds = []
    for images, labels in tqdm(dataloader, total=len(dataloader)):
        out = model(images.cuda())
        pred = out.argmax(dim=1)
        
        targets.append(labels)
        preds.append(pred.detach().cpu())
        
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return targets, preds


# In[62]:

score_tmp = dict()
for key, val in scores.items():
    score_tmp[key] = val[(0, 1000)]
scores = score_tmp
sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}


ablate_groups = []
print (f"Number of sorted_scores key should be 768 - {len(sorted_scores.keys())}")

for i in range(11):
    ablate_groups.append(list(sorted_scores.keys())[:(i+1)*76])

for k, group in enumerate(ablate_groups):
    model = get_fn_model_loader('vit_b_16_timm')(pretrained=True)
    model.eval().cuda();

    targets, preds = evaluate(model, dataloader, neurons_to_ablate=group)
    accuracy = Accuracy(task="multiclass", num_classes=1000)
    acc = accuracy(preds, targets)

    path = f'pred_group_{k:03d}.pkl'
    with open(path, 'wb') as f:
        pickle.dump({
                     'targets': targets,
                     'preds': preds,
                     'acc': acc
                 }, f)

    