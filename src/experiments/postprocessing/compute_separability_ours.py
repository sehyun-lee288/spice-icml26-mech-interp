import os
import argparse
import glob
import gc
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm
from itertools import chain
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from experiments.disentangling.constants import PROJECT_DIR, CHOSEN_NEURONS

BATCH_SIZE = 128
NUM_WORKERS = 2


# ============================================================
# Common functions (used by all compute_separability_*.py files)
# ============================================================

def setup_data_and_models(data_path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(data_path, transform=transform)

    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
    print(f"CLIP model loaded on device: {device}")

    return val_dataset, clip_model, clip_preprocess


def tensor_to_pil(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor * 255).numpy().astype('uint8')
    return Image.fromarray(tensor)


def compute_clip_similarity_matrix(image_lists, clip_model, clip_preprocess, device, batch_size=32):
    all_images = []
    cluster_labels = []

    for cluster_idx, cluster_images in enumerate(image_lists):
        if len(cluster_images) > 0:
            for image in cluster_images:
                all_images.append(image)
                cluster_labels.append(cluster_idx)

    if len(all_images) == 0:
        return np.array([]), [], np.array([])

    embeddings_list = []

    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i:i+batch_size]

            try:
                batch_pil = [tensor_to_pil(img) for img in batch_images]
                batch_preprocessed = torch.stack([clip_preprocess(img) for img in batch_pil]).to(device)

                batch_embeddings = clip_model.encode_image(batch_preprocessed)
                batch_embeddings = F.normalize(batch_embeddings, dim=1)
                embeddings_list.append(batch_embeddings.cpu().numpy())

                del batch_preprocessed, batch_embeddings, batch_pil
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue

    if not embeddings_list:
        return np.array([]), cluster_labels, np.array([])

    embeddings = np.vstack(embeddings_list)
    del embeddings_list
    gc.collect()

    similarity_matrix = np.dot(embeddings, embeddings.T)

    return similarity_matrix, cluster_labels, embeddings


def analyze_cluster_similarity(similarity_matrix, cluster_labels):
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    intra_cluster_similarities = []
    inter_cluster_similarities = []

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 1:
            cluster_sim_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            upper_triangle = np.triu(cluster_sim_matrix, k=1)
            intra_similarities = upper_triangle[upper_triangle > 0]
            intra_cluster_similarities.extend(intra_similarities)

    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            if i < j:
                indices_i = np.where(cluster_labels == cluster_i)[0]
                indices_j = np.where(cluster_labels == cluster_j)[0]
                inter_sim_matrix = similarity_matrix[np.ix_(indices_i, indices_j)]
                inter_cluster_similarities.extend(inter_sim_matrix.flatten())

    if not intra_cluster_similarities or not inter_cluster_similarities:
        return None

    intra_mean = np.mean(intra_cluster_similarities)
    inter_mean = np.mean(inter_cluster_similarities)
    separation_score = intra_mean / inter_mean if inter_mean != 0 else float('inf')

    return {
        'intra_mean': float(intra_mean),
        'inter_mean': float(inter_mean),
        'separation_score': float(separation_score),
    }


def load_images_for_clusters(cluster_indices, val_dataset, device):
    all_indices = sorted(set(chain.from_iterable(cluster_indices)))

    subset = Subset(val_dataset, all_indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    cache = {}
    base = 0
    for images, labels in loader:
        batch_size = len(images)
        for k in range(batch_size):
            orig_idx = all_indices[base + k]
            cache[orig_idx] = (images[k], labels[k])
        base += batch_size

    target_images_lists = []
    total_images = 0

    for idx_list in cluster_indices:
        imgs = []
        for idx in idx_list:
            if idx in cache:
                img, _ = cache[idx]
                imgs.append(img)

        if len(imgs) > 0:
            target_images_lists.append(imgs)
            total_images += len(imgs)

    return target_images_lists, total_images, cache


def compute_separability_for_neuron(neuron_idx, cluster_indices, val_dataset, clip_model, clip_preprocess, device):
    if len(cluster_indices) < 2:
        return None

    target_images_lists, total_images, cache = load_images_for_clusters(
        cluster_indices, val_dataset, device
    )

    print(f"Neuron {neuron_idx}: {len(target_images_lists)} clusters, {total_images} images")

    similarity_matrix, cluster_labels, embeddings = compute_clip_similarity_matrix(
        target_images_lists, clip_model, clip_preprocess, device
    )

    if len(similarity_matrix) == 0:
        return None

    result = analyze_cluster_similarity(similarity_matrix, cluster_labels)

    del cache, target_images_lists, similarity_matrix, embeddings
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def compute_summary_statistics(results, model_name, layer_name, method='ours'):
    valid_results = [r for r in results if r['result'] is not None]
    if valid_results:
        intra_list = [r['result']['intra_mean'] for r in valid_results]
        inter_list = [r['result']['inter_mean'] for r in valid_results]

        summary = {
            'model_name': model_name,
            'layer_name': layer_name,
            'method': method,
            'num_neurons': len(results),
            'num_valid': len(valid_results),
            'intra_mean': float(np.nanmean(intra_list)),
            'inter_mean': float(np.nanmean(inter_list)),
            'separability_ratio': float(np.nanmean(intra_list) / np.nanmean(inter_list)),
            'separability_ratio_per_neuron_mean': float(np.nanmean(np.array(intra_list) / np.array(inter_list))),
            'separability_ratio_per_neuron_std': float(np.nanstd(np.array(intra_list) / np.array(inter_list))),
        }
    else:
        summary = {
            'model_name': model_name,
            'layer_name': layer_name,
            'method': method,
            'num_neurons': len(results),
            'num_valid': 0,
        }
    return summary


def print_summary(summary):
    if summary['num_valid'] > 0:
        print(f"\nSummary:")
        print(f"  Valid neurons: {summary['num_valid']}/{summary['num_neurons']}")
        print(f"  Intra-cluster similarity: {summary['intra_mean']:.4f}")
        print(f"  Inter-cluster similarity: {summary['inter_mean']:.4f}")
        print(f"  Separability ratio: {summary['separability_ratio']:.4f}")
        print(f"  Per-neuron ratio: {summary['separability_ratio_per_neuron_mean']:.4f} +/- {summary['separability_ratio_per_neuron_std']:.4f}")
    else:
        print(f"\nNo valid results")


def save_results(output, output_dir, layer_name):
    os.makedirs(output_dir, exist_ok=True)
    safe_layer = layer_name.replace('.', '_')
    output_file = os.path.join(output_dir, f'separability_{safe_layer}.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


def cleanup_memory(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# Ours-specific functions
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="Compute separability for our clustering method")
    parser.add_argument("--model_name", type=str, default="vit_b_16_timm")
    parser.add_argument("--layer_name", type=str, default="blocks.11")
    parser.add_argument("--neuron_indices", type=int, nargs='+', default=None)
    parser.add_argument("--data_path", type=str, default="/project/data/external/ILSVRC/Data/CLS-LOC/val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_ours_clustering(cluster_file):
    with open(cluster_file, 'rb') as f:
        groups = pickle.load(f)
    return groups


def process_neuron_ours(neuron_idx, groups, val_dataset, clip_model, clip_preprocess, device):
    cluster_indices = []
    for cluster_id in range(len(groups)):
        sample_indices = groups[cluster_id]
        if isinstance(sample_indices, np.ndarray):
            sample_indices = sample_indices.tolist()
        if len(sample_indices) > 0:
            cluster_indices.append(sample_indices)

    return compute_separability_for_neuron(
        neuron_idx, cluster_indices, val_dataset, clip_model, clip_preprocess, device
    )


def main():
    args = get_args()

    val_dataset, clip_model, clip_preprocess = setup_data_and_models(args.data_path, args.device)

    cluster_dir = f'{PROJECT_DIR}/results/clustering/kmeans_efficient/{args.model_name}/{args.layer_name}/None'

    if args.neuron_indices:
        neuron_indices = args.neuron_indices
    else:
        neuron_indices = CHOSEN_NEURONS.get(args.model_name, {}).get(args.layer_name, [])
        if not neuron_indices:
            neuron_dirs = sorted(glob.glob(os.path.join(cluster_dir, '*')))
            neuron_indices = [int(os.path.basename(d)) for d in neuron_dirs if os.path.isdir(d)]

    results = []
    for neuron_idx in tqdm(neuron_indices, desc="Processing neurons"):
        cluster_file = f'{cluster_dir}/{neuron_idx:04d}/idx_00000_00100.pkl'

        if not os.path.exists(cluster_file):
            print(f"Cluster file not found: {cluster_file}")
            results.append({'neuron_idx': neuron_idx, 'result': None})
            continue

        groups = load_ours_clustering(cluster_file)
        result = process_neuron_ours(neuron_idx, groups, val_dataset, clip_model, clip_preprocess, args.device)

        results.append({
            'neuron_idx': neuron_idx,
            'num_clusters': len(groups),
            'result': result,
        })

        if neuron_idx % 5 == 0:
            cleanup_memory(args.device)

    summary = compute_summary_statistics(results, args.model_name, args.layer_name, method='ours')
    print_summary(summary)

    output_dir = args.output_dir or f'{PROJECT_DIR}/results/separability/ours/{args.model_name}'
    output = {'summary': summary, 'results': results}
    save_results(output, output_dir, args.layer_name)


if __name__ == "__main__":
    main()
