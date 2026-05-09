import os
import argparse
import pickle
import random
from tqdm.auto import tqdm
import json

import torch
from models import get_fn_model_loader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from experiments.disentangling.attribution import get_layer_by_name

from dsets import get_dataset
from contextlib import contextmanager

import logging

logging.basicConfig(level=logging.WARNING)


def ablate_and_collect(neurons_to_ablate, subset, model_name, layer_name, random_sample_indices):
    """
    Ablate the specified neurons for each image in the subset and collect logits, predictions, and labels.

    Args:
        neurons_to_ablate (list or tensor): Indices of neurons to ablate.
        subset (iterable): Iterable of (img, label) pairs.
        model_name (str): Name of the model to load.
        layer_name (str): Name of the layer to ablate.
        random_sample_indices (list or tensor): Indices of the random samples.

    Returns:
        dict: Dictionary containing logits, preds, labels, and random_sample_indices.
    """
    ours_logits = []
    ours_preds = []
    ours_labels = []

    model = get_fn_model_loader(model_name)().eval().cuda()

    @contextmanager
    def forward_hook_context(module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    for img, label in tqdm(subset, total=len(subset)):
        def zero_hook(module, input, output):
            mask = torch.ones_like(output)
            if mask.ndim == 4:
                mask[:, neurons_to_ablate, :, :] = 0
            elif mask.ndim == 3:
                mask[:, :, neurons_to_ablate] = 0
            else:
                raise Exception(f"Not implemented, mask.ndim = {mask.ndim}")
            return output * mask

        layer = get_layer_by_name(model, layer_name)
        with forward_hook_context(layer, zero_hook):
            logit = model(img.unsqueeze(0).cuda()).detach().cpu()
            pred_label = model(img.unsqueeze(0).cuda()).argmax().item()

        ours_logits.append(logit)
        ours_preds.append(pred_label)
        ours_labels.append(label)

    ours_logits = torch.cat(ours_logits, dim=0)
    ours_labels = torch.tensor(ours_labels)
    ours_preds = torch.tensor(ours_preds)

    accuracy = (ours_preds == ours_labels).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")

    return ours_logits, ours_preds, ours_labels, random_sample_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation experiment for different models and layers"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to run ablation for (if not set, runs all in list)",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default=None,
        help="rank or num_cluster",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=20,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_layer_dict = {
        'vit_b_16_timm': 'blocks.11',
        'resnet50_timm': 'layer4.2',
        'resnet34_timm': 'layer4.2',
        'vit_s_16_timm': 'blocks.11',
        'convnext_timm': 'stages.3.blocks.2',
        'vgg19_timm': 'features.36',
    }
    
    model_last_dim_dict = {
        'vit_b_16_timm': 768,
        'resnet50_timm': 2048,
        'resnet34_timm': 512,
        'vit_s_16_timm': 384,
        'convnext_timm': 1024,
        'vgg19_timm': 512
    }

    dataset_name = 'imagenet'
    data_path = '/project/data/external/ILSVRC/Data/CLS-LOC'

    with open(
        "/project/results/ablation/resnet50_timm/ours_bottom_logits_preds_labels_random_samples.pkl",
        "rb",
    ) as f:
        save_data = pickle.load(f)
    random_sample_indices = save_data['random_sample_indices']

    model_name = args.model_name
    layer_name = model_layer_dict[model_name]
    
    # Make dataset
    model = get_fn_model_loader(model_name)().eval().cuda()
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    transform = transform.transforms
    dataset = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=True,  # Get original images
        split="val",
        transform=transform,
    )
    subset = torch.utils.data.Subset(dataset, random_sample_indices)

    # ==========    Random run    ========== #
    @contextmanager
    def forward_hook_context(module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    num_last_dim = model_last_dim_dict[model_name]
    partition_size = num_last_dim // 20
    neuron_list = [
        list(range(i * partition_size, (i + 1) * partition_size))
        for i in range(20)
    ]
    random.shuffle(neuron_list)
        
    model = get_fn_model_loader(model_name)().eval().cuda()

    zero_indices = []
    result_dict = dict()
    for i, neurons in enumerate(neuron_list):
        zero_indices.extend(neurons)
        
        random_logits = []
        random_preds = []
        random_labels = []
        for img, label in tqdm(subset, total=len(subset)):
            def zero_channels_hook(zero_indices):
                def hook(module, input, output):
                    mask = torch.ones_like(output)
                    if mask.ndim == 4:
                        mask[:, zero_indices, :, :] = 0
                    elif mask.ndim == 3:
                        mask[:, :, zero_indices] = 0
                    else:
                        raise Exception(f"Not implemented, mask.ndim = {mask.ndim}")
                    return output * mask
                return hook

            layer = get_layer_by_name(model, layer_name)
            with forward_hook_context(layer, zero_channels_hook(zero_indices)):
                logit = model(img.unsqueeze(0).cuda()).detach().cpu()
                pred_label = model(img.unsqueeze(0).cuda()).argmax().item()

            random_logits.append(logit)
            random_preds.append(pred_label)
            random_labels.append(label)

        random_logits = torch.cat(random_logits, dim=0)
        random_labels = torch.tensor(random_labels)
        random_preds = torch.tensor(random_preds)
        
        # Compute accuracy
        accuracy = (random_preds == random_labels).float().mean().item()
        print(f"Random Accuracy: {accuracy:.4f}")
        
        result_dict[i] = (random_logits, random_labels, random_preds)
        
    with open(
        f"/project/results/ablation/{model_name}/random_partition_result_dict.pkl",
        "wb",
    ) as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    main()