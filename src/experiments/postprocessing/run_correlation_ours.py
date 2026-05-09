import os
import argparse
import glob
import json
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import clip
from tqdm import trange
from safetensors.torch import load_file

from models import get_fn_model_loader
from experiments.preprocessing.compute_top_activations import aggregate_spatial_dimensions
from experiments.disentangling.constants import PROJECT_DIR
from experiments.postprocessing.run_AUC_MAD_ours import create_cluster_grid, generate_captions

"""
python run_correlation_ours.py \
    --model_name vit_b_16_timm \
    --layer_name blocks.11 \
    --neuron_idx 360
"""

def get_args():
    parser = argparse.ArgumentParser(description="Run correlation evaluation for our clustering method")
    parser.add_argument("--model_name", type=str, default="vit_b_16_timm")
    parser.add_argument("--layer_name", type=str, default="blocks.11")
    parser.add_argument("--neuron_idx", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--skip_crop", action="store_true")
    parser.add_argument("--skip_caption", action="store_true")
    parser.add_argument("--caption_save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_activations(model_name, layer_name, device):
    safe_layer = layer_name.replace('.', '_')
    activation_dir = f'{PROJECT_DIR}/results/activations/imagenet'
    act_path = f'{activation_dir}/{model_name}/activations_{safe_layer}_output_raw.safetensors'

    act = load_file(act_path)[layer_name]
    model_type = 'vit' if act.ndim == 3 else 'conv'
    act = aggregate_spatial_dimensions(act, aggregation="top_mean", top_percentile=10.0, type=model_type)
    return act.to(device)


def compute_text_embeddings(concepts, clip_model, device):
    text_tokens = clip.tokenize(concepts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=1)
    return text_features


def compute_image_embeddings(sample_indices, val_dataset, clip_model, device, batch_size=32):
    from torch.utils.data import DataLoader, Subset

    subset = Subset(val_dataset, sample_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings_list = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = F.normalize(features, dim=1)
            embeddings_list.append(features.cpu())

    return torch.cat(embeddings_list, dim=0)


def simulate_correlations(args, groups, caption_save_dir):
    """
    Evaluates cluster quality via correlation between neuron activations
    and CLIP text embeddings of cluster concepts.
    """
    from torchvision import datasets

    # Load CLIP model and preprocessing
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=args.device, jit=False)
    clip_model.eval()

    # Load activations
    activations = load_activations(args.model_name, args.layer_name, args.device)
    neuron_activations = activations[:, args.neuron_idx]

    # Load dataset with CLIP preprocessing
    val_dataset = datasets.ImageFolder(
        '/project/data/external/ILSVRC/Data/CLS-LOC/val',
        transform=clip_preprocess
    )

    cluster_results = []

    for cluster_idx in trange(len(groups), desc="Computing correlations"):
        sample_indices = groups[cluster_idx]
        if isinstance(sample_indices, np.ndarray):
            sample_indices = sample_indices.tolist()
        cluster_size = len(sample_indices)

        # Load concepts from response file
        resp_pattern = f'{caption_save_dir}/concept_response_cluster_{cluster_idx:03d}_*.json'
        resp_candidates = glob.glob(resp_pattern)

        if not resp_candidates:
            print(f"No response file for cluster {cluster_idx}")
            cluster_results.append({
                'cluster_idx': cluster_idx,
                'cluster_size': cluster_size,
                'sample_indices': sample_indices,
                'concepts': None,
                'correlation': None,
            })
            continue

        with open(resp_candidates[0], "r") as f:
            resp = json.load(f)

        concepts = resp.get('step1_common', [])
        if not concepts:
            cluster_results.append({
                'cluster_idx': cluster_idx,
                'cluster_size': cluster_size,
                'sample_indices': sample_indices,
                'concepts': [],
                'correlation': None,
            })
            continue

        # Compute text embedding for concepts
        concept_text = ', '.join(concepts)
        text_embedding = compute_text_embeddings([concept_text], clip_model, args.device)

        # Get neuron activations for samples in this cluster
        cluster_activations = neuron_activations[sample_indices]

        # Compute CLIP image embeddings for cluster samples
        image_embeddings = compute_image_embeddings(
            sample_indices, val_dataset, clip_model, args.device
        )

        # Compute similarity between images and concept text
        with torch.no_grad():
            image_embeddings = image_embeddings.to(args.device)
            clip_similarities = (image_embeddings @ text_embedding.T).squeeze(-1)

        # Compute correlation between neuron activations and CLIP similarities
        cluster_activations_cpu = cluster_activations.cpu().float()
        clip_similarities_cpu = clip_similarities.cpu().float()

        if cluster_activations_cpu.std() > 0 and clip_similarities_cpu.std() > 0:
            combined = torch.stack([cluster_activations_cpu, clip_similarities_cpu], dim=0)
            corr_matrix = torch.corrcoef(combined)
            correlation = float(corr_matrix[0, 1].item())
        else:
            correlation = 0.0

        print(f"Cluster {cluster_idx}: concepts={concepts}, correlation={correlation:.4f}")

        cluster_results.append({
            'cluster_idx': cluster_idx,
            'cluster_size': cluster_size,
            'sample_indices': sample_indices,
            'concepts': concepts,
            'concept_text': concept_text,
            'correlation': correlation,
        })

    # Compute summary statistics
    valid_correlations = [r['correlation'] for r in cluster_results if r['correlation'] is not None]
    summary = {
        'model_name': args.model_name,
        'layer_name': args.layer_name,
        'neuron_idx': args.neuron_idx,
        'num_clusters': len(groups),
        'num_valid': len(valid_correlations),
        'mean_correlation': float(np.mean(valid_correlations)) if valid_correlations else None,
        'std_correlation': float(np.std(valid_correlations)) if valid_correlations else None,
    }

    results = {
        'summary': summary,
        'clusters': cluster_results,
    }

    return results


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = get_fn_model_loader(args.model_name)()

    cluster_dir = f'{PROJECT_DIR}/results/clustering/kmeans_efficient'
    cluster_file = f'{cluster_dir}/{args.model_name}/{args.layer_name}/None/{args.neuron_idx:04d}/idx_00000_00100.pkl'
    with open(cluster_file, 'rb') as f:
        groups = pickle.load(f)

    output_dir = args.output_dir or f'{PROJECT_DIR}/results/correlation/ours'
    save_dir = f'{output_dir}/{args.model_name}/{args.layer_name}/{args.neuron_idx:04d}'
    os.makedirs(save_dir, exist_ok=True)

    caption_save_dir = args.caption_save_dir or f'{PROJECT_DIR}/results/caption/ours/{args.model_name}/{args.layer_name}/{args.neuron_idx:04d}'
    os.makedirs(caption_save_dir, exist_ok=True)

    num_clusters = len(groups)

    if not args.skip_crop:
        for cluster_idx in trange(num_clusters, desc="Creating cluster grids"):
            sample_indices = groups[cluster_idx]
            save_path = f"{caption_save_dir}/cluster_{cluster_idx:03d}_size_{len(sample_indices)}.png"
            if os.path.exists(save_path):
                continue
            create_cluster_grid(sample_indices, args.neuron_idx, model, args.layer_name, args.model_name, save_path)

    if not args.skip_caption:
        from openai import OpenAI
        api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Use --openai_api_key or set OPENAI_API_KEY env var")
        client = OpenAI(api_key=api_key)
        generate_captions(caption_save_dir, client)

    # Run correlation evaluation
    results_path = os.path.join(save_dir, 'correlation_results.json')

    if not os.path.exists(results_path):
        results = simulate_correlations(args, groups, caption_save_dir)
        # Save results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"\nResults already been there. Loaded from {results_path}")

    # Print summary
    summary = results['summary']
    if summary['mean_correlation'] is not None:
        print(f"\nSummary: {summary['num_valid']} clusters evaluated")
        print(f"Mean correlation: {summary['mean_correlation']:.4f} +/- {summary['std_correlation']:.4f}")


if __name__ == "__main__":
    main()
