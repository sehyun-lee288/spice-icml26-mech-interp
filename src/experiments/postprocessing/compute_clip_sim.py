import os
import argparse
import pickle
import gc
import random
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from experiments.disentangling.constants import PROJECT_DIR

MODEL_CONFIGS = {
    'convnext_timm': {
        'dims': [256, 512, 1024],
        'layers': ['stages.1', 'stages.2', 'stages.3']
    },
    'resnet50_timm': {
        'dims': [512, 1024, 2048],
        'layers': ['layer2.1', 'layer3.3', 'layer4.2']
    },
    'vit_b_16_timm': {
        'dims': [768, 768, 768],
        'layers': ['blocks.2', 'blocks.6', 'blocks.11']
    },
    'densenet_timm': {
        'dims': [32, 32, 32],
        'layers': ["features.denseblock2.denselayer1", "features.denseblock3.denselayer12", "features.denseblock4.denselayer16"]
    },
    'clip_vit_b_16_timm': {
        'dims': [768, 768, 768],
        'layers': ['blocks.2', 'blocks.6', 'blocks.11']
    }
}

CHOSEN_NEURONS = {
    'resnet50_timm': {
        'layer2.1': [289, 493, 295, 360, 7, 413, 74, 35, 459, 108, 430, 71, 240, 383, 505, 218, 60, 235, 36, 93, 462, 0, 277, 276, 158, 428, 416, 244, 366, 238, 76, 189, 496, 453, 142, 114, 412, 79, 16, 88, 300, 354, 214, 484, 440, 338, 409, 423, 356, 111, 274, 376, 234, 241, 1, 4, 95, 498, 282, 404, 131, 415, 242, 508, 301, 507, 483, 84, 395, 217, 369, 61, 110, 22, 143, 147, 476, 50, 155, 275, 355, 8, 222, 89, 26, 42, 458, 267, 293, 55, 85, 500, 441, 19, 286, 253, 205, 92, 374, 427],
        'layer3.3': [439, 129, 246, 603, 398, 193, 147, 785, 308, 819, 835, 447, 211, 339, 466, 288, 252, 3, 460, 757, 600, 571, 173, 540, 276, 86, 393, 629, 111, 529, 864, 89, 221, 553, 354, 293, 993, 922, 581, 64, 358, 806, 284, 433, 237, 263, 234, 42, 525, 66, 296, 729, 507, 65, 318, 422, 188, 572, 459, 509, 977, 336, 116, 178, 462, 121, 915, 935, 418, 132, 326, 655, 509, 292, 713, 935, 950, 321, 582, 927, 817, 390, 703, 300, 537, 273, 196, 778, 215, 829, 882, 464, 687, 28, 1009, 305, 528, 208, 678, 23],
        'layer4.2': [457, 1451, 685, 1752, 231, 928, 1764, 1391, 861, 32, 1484, 981, 299, 2023, 145, 1071, 387, 817, 932, 530, 709, 1544, 572, 1812, 410, 1073, 1818, 1568, 957, 1024, 1655, 1884, 87, 1549, 1813, 1429, 69, 1447, 782, 1299, 629, 1695, 1118, 1774, 1667, 684, 1269, 487, 875, 1535, 1734, 1353, 1194, 239, 886, 1769, 1233, 21, 1949, 41, 1826, 371, 1631, 190, 591, 1777, 157, 838, 1159, 965, 1112, 719, 1497, 380, 1387, 660, 929, 1341, 2018, 529, 558, 334, 416, 1407, 682, 914, 1951, 1331, 1720, 1424, 1254, 1843, 489, 1325, 664, 27, 378, 1078, 930, 174],
    },
    'vit_b_16_timm': {
        'blocks.2': [294, 27, 276, 523, 703, 97, 512, 669, 151, 442, 679, 641, 220, 10, 322, 701, 235, 572, 544, 123, 707, 30, 552, 287, 95, 158, 72, 745, 390, 260, 232, 433, 242, 439, 522, 613, 167, 226, 476, 397, 598, 603, 273, 616, 204, 730, 419, 738, 457, 47, 532, 367, 591, 191, 239, 693, 415, 358, 709, 105, 497, 75, 647, 500, 325, 680, 172, 57, 574, 622, 606, 272, 644, 184, 764, 531, 300, 338, 743, 464, 140, 237, 122, 174, 29, 87, 681, 243, 279, 541, 431, 408, 486, 141, 344, 265, 393, 161, 23, 350],
        'blocks.6': [288, 672, 714, 31, 315, 756, 682, 284, 740, 687, 147, 658, 757, 545, 441, 370, 600, 545, 99, 291, 59, 183, 608, 499, 388, 688, 153, 121, 50, 670, 231, 186, 732, 576, 590, 161, 64, 539, 85, 331, 489, 626, 746, 733, 488, 211, 216, 1, 102, 280, 357, 516, 463, 747, 761, 654, 674, 513, 387, 298, 76, 457, 315, 389, 299, 289, 476, 607, 136, 749, 503, 399, 704, 164, 230, 584, 120, 411, 478, 127, 135, 52, 293, 67, 184, 677, 632, 23, 181, 128, 196, 485, 721, 453, 367, 175, 202, 91, 220, 165],
        'blocks.11': [19, 748, 472, 424, 169, 349, 521, 396, 236, 30, 42, 220, 251, 615, 154, 460, 183, 410, 567, 247, 162, 205, 494, 125, 226, 654, 87, 291, 712, 294, 88, 422, 57, 516, 202, 256, 172, 0, 111, 338, 179, 449, 660, 26, 322, 278, 75, 560, 337, 245, 599, 608, 292, 470, 594, 240, 29, 388, 5, 697, 184, 345, 207, 231, 310, 691, 657, 720, 489, 589, 746, 147, 237, 330, 559, 3, 38, 121, 355, 537, 526, 717, 616, 627, 766, 667, 22, 164, 448, 738, 762, 702, 201, 760, 78, 139, 757, 392, 765, 404],
    },
    'clip_vit_b_16_timm': {
        'blocks.2': [294, 27, 276, 523, 703, 97, 512, 669, 151, 442, 679, 641, 220, 10, 322, 701, 235, 572, 544, 123, 707, 30, 552, 287, 95, 158, 72, 745, 390, 260, 232, 433, 242, 439, 522, 613, 167, 226, 476, 397, 598, 603, 273, 616, 204, 730, 419, 738, 457, 47, 532, 367, 591, 191, 239, 693, 415, 358, 709, 105, 497, 75, 647, 500, 325, 680, 172, 57, 574, 622, 606, 272, 644, 184, 764, 531, 300, 338, 743, 464, 140, 237, 122, 174, 29, 87, 681, 243, 279, 541, 431, 408, 486, 141, 344, 265, 393, 161, 23, 350],
        'blocks.6': [288, 672, 714, 31, 315, 756, 682, 284, 740, 687, 147, 658, 757, 545, 441, 370, 600, 545, 99, 291, 59, 183, 608, 499, 388, 688, 153, 121, 50, 670, 231, 186, 732, 576, 590, 161, 64, 539, 85, 331, 489, 626, 746, 733, 488, 211, 216, 1, 102, 280, 357, 516, 463, 747, 761, 654, 674, 513, 387, 298, 76, 457, 315, 389, 299, 289, 476, 607, 136, 749, 503, 399, 704, 164, 230, 584, 120, 411, 478, 127, 135, 52, 293, 67, 184, 677, 632, 23, 181, 128, 196, 485, 721, 453, 367, 175, 202, 91, 220, 165],
        'blocks.11': [19, 748, 472, 424, 169, 349, 521, 396, 236, 30, 42, 220, 251, 615, 154, 460, 183, 410, 567, 247, 162, 205, 494, 125, 226, 654, 87, 291, 712, 294, 88, 422, 57, 516, 202, 256, 172, 0, 111, 338, 179, 449, 660, 26, 322, 278, 75, 560, 337, 245, 599, 608, 292, 470, 594, 240, 29, 388, 5, 697, 184, 345, 207, 231, 310, 691, 657, 720, 489, 589, 746, 147, 237, 330, 559, 3, 38, 121, 355, 537, 526, 717, 616, 627, 766, 667, 22, 164, 448, 738, 762, 702, 201, 760, 78, 139, 757, 392, 765, 404],
    },
}


def get_args():
    parser = argparse.ArgumentParser(description='Compute cluster similarities using CLIP embeddings')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--layer_idx', type=int, default=None)
    parser.add_argument('--neuron_range', type=str, default=None, help='e.g., "0,100" for neurons 0-99')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--efficient', action='store_true', help='Select only 10% of neurons')
    return parser.parse_args()


def setup_data_and_models():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dir = '/project/data/external/ILSVRC/Data/CLS-LOC/val'
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)

    return val_dataset, clip_model, clip_preprocess, device


def tensor_to_pil(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = torch.clamp(tensor * std + mean, 0, 1)
    tensor = (tensor.permute(1, 2, 0) * 255).numpy().astype('uint8')
    return Image.fromarray(tensor)


def compute_clip_similarity_matrix(image_lists, clip_model, clip_preprocess, device, batch_size=32):
    all_images, cluster_labels = [], []

    for cluster_idx, cluster_images in enumerate(image_lists):
        for image in cluster_images:
            all_images.append(image)
            cluster_labels.append(cluster_idx)

    if not all_images:
        return np.array([]), [], np.array([])

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i:i + batch_size]
            try:
                batch_pil = [tensor_to_pil(img) for img in batch_images]
                batch_preprocessed = torch.stack([clip_preprocess(img) for img in batch_pil]).to(device)
                batch_embeddings = clip_model.encode_image(batch_preprocessed)
                batch_embeddings = F.normalize(batch_embeddings, dim=1)
                embeddings_list.append(batch_embeddings.cpu().numpy())

                del batch_preprocessed, batch_embeddings
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing batch: {e}")

    if not embeddings_list:
        return np.array([]), cluster_labels, np.array([])

    embeddings = np.vstack(embeddings_list)
    similarity_matrix = np.dot(embeddings, embeddings.T)

    return similarity_matrix, cluster_labels, embeddings


def analyze_cluster_similarity(similarity_matrix, cluster_labels, neuron_idx=None, is_plot=False):
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    intra_sims, inter_sims = [], []
    per_cluster_stats, per_inter_stats = {}, []

    for cluster_id in unique_clusters:
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) > 1:
            cluster_sim = similarity_matrix[np.ix_(indices, indices)]
            upper_tri = np.triu(cluster_sim, k=1)
            intra = upper_tri[upper_tri > 0]
            intra_sims.extend(intra)

            per_cluster_stats[int(cluster_id)] = {
                "neuron_idx": neuron_idx,
                "cluster_id": int(cluster_id),
                "size": len(indices),
                "intra_sims": intra.copy(),
                "intra_mean": float(np.mean(intra)) if len(intra) > 0 else None,
            }

    for i, ci in enumerate(unique_clusters):
        for j, cj in enumerate(unique_clusters):
            if i < j:
                idx_i = np.where(cluster_labels == ci)[0]
                idx_j = np.where(cluster_labels == cj)[0]
                inter = similarity_matrix[np.ix_(idx_i, idx_j)].flatten()
                if len(inter) > 0:
                    inter_sims.extend(inter)
                    per_inter_stats.append({
                        "neuron_idx": neuron_idx,
                        "cluster_i": int(ci),
                        "cluster_j": int(cj),
                        "inter_sims": inter.copy(),
                        "inter_mean": float(np.mean(inter)),
                    })

    if is_plot and intra_sims and inter_sims:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(intra_sims, bins=50, alpha=0.7, label='Intra', color='blue')
        axes[0].hist(inter_sims, bins=50, alpha=0.7, label='Inter', color='red')
        axes[0].legend()
        axes[1].boxplot([intra_sims, inter_sims], labels=['Intra', 'Inter'])
        plt.tight_layout()
        plt.show()

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0

    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'separation_score': intra_mean / inter_mean if inter_mean != 0 else float('inf'),
        'intra_cluster_similarities': intra_sims,
        'inter_cluster_similarities': inter_sims,
        'per_cluster_stats': per_cluster_stats,
        'per_inter_stats': per_inter_stats,
    }


def process_single_neuron(args_tuple):
    model_name, layer, neuron_idx, val_dataset, clip_model, clip_preprocess, device = args_tuple

    clustering_path = f"{PROJECT_DIR}/results/clustering/kmeans_efficient/{model_name}/{layer}/None/{neuron_idx:04d}/idx_00000_00100.pkl"

    if not os.path.exists(clustering_path):
        return None

    try:
        with open(clustering_path, "rb") as f:
            cluster_indices = pickle.load(f)

        if not cluster_indices:
            return None

        all_indices = sorted(set(chain.from_iterable(cluster_indices)))
        if not all_indices:
            return None

        subset = Subset(val_dataset, all_indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

        cache = {}
        base = 0
        for images, labels in loader:
            for k in range(len(images)):
                cache[all_indices[base + k]] = (images[k], labels[k])
            base += len(images)

        image_lists = []
        for idx_list in cluster_indices:
            imgs = [cache[idx][0] for idx in idx_list if idx in cache]
            if imgs:
                image_lists.append(imgs)

        if not image_lists:
            return None

        similarity_matrix, cluster_labels, embeddings = compute_clip_similarity_matrix(
            image_lists, clip_model, clip_preprocess, device
        )

        if len(similarity_matrix) == 0:
            return None

        result = analyze_cluster_similarity(similarity_matrix, cluster_labels, neuron_idx=neuron_idx)

        del cache, image_lists, similarity_matrix, embeddings
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Error processing neuron {neuron_idx}: {e}")
        return None


def main():
    args = get_args()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {args.model_name}")

    config = MODEL_CONFIGS[args.model_name]
    layers = config['layers']
    dims = config['dims']

    val_dataset, clip_model, clip_preprocess, device = setup_data_and_models()

    layer_indices = [args.layer_idx] if args.layer_idx is not None else range(len(layers))

    for i in layer_indices:
        if i >= len(layers):
            continue

        layer = layers[i]
        dim = dims[i]

        output_dir = f"{PROJECT_DIR}/results/similarities/{args.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/cluster_similarity_{layer}.pkl"

        if args.skip_existing and os.path.exists(output_file):
            print(f"Skipping layer {layer} (exists)")
            continue

        if args.neuron_range:
            start, end = map(int, args.neuron_range.split(','))
            neuron_range = list(range(start, min(end, dim)))
        elif args.model_name in CHOSEN_NEURONS and layer in CHOSEN_NEURONS[args.model_name]:
            neuron_range = CHOSEN_NEURONS[args.model_name][layer]
        else:
            neuron_range = list(range(dim))

        if args.efficient:
            num_select = max(1, int(len(neuron_range) * 0.1))
            neuron_range = sorted(random.sample(neuron_range, num_select))
            print(f"Efficient mode: {num_select} neurons selected")

        print(f"Processing layer {layer}: {len(neuron_range)} neurons")

        results = []
        for neuron_idx in tqdm(neuron_range, desc=f"Layer {layer}"):
            result = process_single_neuron((
                args.model_name, layer, neuron_idx, val_dataset,
                clip_model, clip_preprocess, device
            ))
            results.append(result)

            if neuron_idx % 10 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        valid = len([r for r in results if r is not None])
        print(f"Layer {layer} complete: {valid}/{len(results)} neurons")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
