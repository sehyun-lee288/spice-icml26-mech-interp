import os
import argparse
import pickle
import json
from tqdm.auto import tqdm

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

    save_data = {
        "logits": ours_logits,
        "preds": ours_preds,
        "labels": ours_labels,
        "random_sample_indices": random_sample_indices,
    }

    return save_data


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

    # Load num clusters
    with open(
        f"/project/results/stats/num_concept_clusters_raw_{model_name}.json", "r"
    ) as f:
        rank_list = json.load(f)  
    last_layer_rank = rank_list[list(rank_list.keys())[-1]]
    sorted_scores = sorted(
        zip(range(len(last_layer_rank)), last_layer_rank), key=lambda x: x[1]
    )

    # Set experiment group
    bottom_neurons = sorted_scores[:100]
    top_neurons = sorted_scores[-100:]
    middle_neurons = sorted_scores[950:1050]

    top_neurons = [n[0] for n in top_neurons]
    middle_neurons = [n[0] for n in middle_neurons]
    bottom_neurons = [n[0] for n in bottom_neurons]

    # ==========    Random run    ========== #
    random_logits = []
    random_preds = []
    random_labels = []
    
    model = get_fn_model_loader(model_name)().eval().cuda()


    @contextmanager
    def forward_hook_context(module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    for img, label in tqdm(subset, total=len(subset)):
        def random_zero_hook(module, input, output):
            # output: (batch, channels, H, W)
            # Randomly select 100 unique channels (neurons) to zero out
            num_channels = output.shape[1]
            num_to_zero = min(100, num_channels)
            zero_indices = torch.randperm(num_channels)[:num_to_zero]
            mask = torch.ones_like(output)
            if mask.ndim == 4:
                mask[:, zero_indices, :, :] = 0
            elif mask.ndim == 3:
                mask[:, :, zero_indices] = 0
            else:
                raise Exception(f"Not implemented, mask.ndim = {mask.ndim}")
            return output * mask

        layer = get_layer_by_name(model, layer_name)
        with forward_hook_context(layer, random_zero_hook):
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

    save_data = {
        "logits": random_logits,
        "preds": random_preds,
        "labels": random_labels,
        "random_sample_indices": random_sample_indices,
    }

    with open(
        f"/project/results/ablation/{model_name}/num_cluster_random_logits_preds_labels_random_samples.pkl",
        "wb",
    ) as f:
        pickle.dump(save_data, f)

    # ==========    Ours run    ========== #
    save_data = ablate_and_collect(
        top_neurons, subset, model_name, layer_name, random_sample_indices
    )
    with open(
        f"/project/results/ablation/{model_name}/num_cluster_ours_top_logits_preds_labels_random_samples.pkl",
        "wb",
    ) as f:
        pickle.dump(save_data, f)

    save_data = ablate_and_collect(
        bottom_neurons, subset, model_name, layer_name, random_sample_indices
    )
    with open(
        f"/project/results/ablation/{model_name}/num_cluster_ours_bottom_logits_preds_labels_random_samples.pkl",
        "wb",
    ) as f:
        pickle.dump(save_data, f)


if __name__ == "__main__":
    main()