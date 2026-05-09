import os
import argparse
import glob
import json
import base64
import pickle
import random
import io

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from google import genai
from google.genai import types
from ratelimit import limits, sleep_and_retry

from models import get_fn_model_loader
from experiments.preprocessing.crop_activation_regions import get_cropped_images
from experiments.preprocessing.extract_activations import ActivationExtractor
from experiments.preprocessing.compute_top_activations import aggregate_spatial_dimensions
from experiments.disentangling.constants import PROJECT_DIR

CALLS_PER_MINUTE = 100
PERIOD_IN_SECONDS = 60 # For RPM

class ImageDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def get_args():
    parser = argparse.ArgumentParser(description="Run AUC/MAD evaluation using Gemini for captioning and image generation")
    parser.add_argument("--model_name", type=str, default="vit_b_16_timm")
    parser.add_argument("--layer_name", type=str, default="blocks.11")
    parser.add_argument("--neuron_idx", type=int, default=360)
    parser.add_argument("--n_images", type=int, default=3, help="Number of images to synthesize per cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--skip_crop", action="store_true")
    parser.add_argument("--skip_caption", action="store_true")
    parser.add_argument("--skip_synthesis", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def create_cluster_grid(sample_indices, neuron_idx, model, layer_name, model_name, save_path):
    """Create a grid of cropped activation regions for a cluster."""
    import matplotlib.pyplot as plt

    cropped_results = get_cropped_images(
        sample_indices=sample_indices,
        neuron_idx=neuron_idx,
        model=model,
        layer_name=layer_name,
        config_file=f"{PROJECT_DIR}/src/configs/imagenet/{model_name}.yaml",
        crop_method="threshold",
        batch_size=8,
        device="cuda"
    )
    imgs = [results[0] for results in cropped_results]

    fig, axes = plt.subplots(1, len(sample_indices), dpi=200)
    plt.subplots_adjust(hspace=0, wspace=0)
    if len(sample_indices) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    fig.savefig(save_path, dpi=200)
    plt.close()

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD_IN_SECONDS)
def generate_caption_with_gemini(image_path, client):
    """Use Gemini to describe common features in the cluster image grid."""
    image = Image.open(image_path)

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    prompt = """Look at this image grid showing cropped regions from neural network activations.
Describe a common caption that captures what is shared across all these image regions.
Focus on:
1. Common objects, textures, or patterns
2. Shared colors or visual characteristics
3. Any repeated elements or themes

Provide a concise description (3-5 words) of what these images have in common.
Only output the common caption, nothing else."""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_bytes(data=img_byte_arr, mime_type='image/png'),
                    types.Part.from_text(text=prompt),
                ]
            )
        ]
    )

    return response.text.strip()


def generate_captions(save_dir, client):
    """Generate captions for all cluster grids using Gemini."""
    flist = sorted(glob.glob(f'{save_dir}/cluster_*_size_*.png'))

    for fname in flist:
        # Extract cluster_idx and size from filename
        import re
        match = re.search(r'cluster_(\d+)', fname)
        cluster_idx = int(match.group(1)) if match else None

        match = re.search(r'size_(\d+)', fname)
        size = int(match.group(1)) if match else None

        out_fname = f'{save_dir}/gemini_caption_cluster_{cluster_idx:03d}_size_{size}.json'
        if os.path.exists(out_fname):
            continue

        try:
            caption = generate_caption_with_gemini(fname, client)
            print(f"Cluster {cluster_idx}: {caption}")

            resp = {
                'cluster_idx': cluster_idx,
                'size': size,
                'caption': caption,
            }
            with open(out_fname, 'w', encoding='utf-8') as f:
                json.dump(resp, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error generating caption for cluster {cluster_idx}: {e}")
            continue

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD_IN_SECONDS)
def generate_image_with_gemini(prompt, out_path, client):
    """Generate a single image using Gemini's native image generation."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents=f"Generate a realistic photo image of: {prompt}",
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
            )
        )

        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    image_data = part.inline_data.data
                    image = Image.open(io.BytesIO(image_data))
                    image.save(out_path)
                    return True
        return False

    except Exception as e:
        print(f"    Gemini image generation error: {e}")
        return False


def synthesize_images(save_dir, num_clusters, n_images, client):
    """Synthesize images using Gemini for each cluster."""
    prompt_prefix = "realistic photo of a close up of"

    for cluster_idx in tqdm(range(num_clusters), desc="Synthesizing images"):
        # Load caption
        caption_pattern = f'{save_dir}/gemini_caption_cluster_{cluster_idx:03d}_*.json'
        caption_candidates = glob.glob(caption_pattern)
        if not caption_candidates:
            print(f"No caption file for cluster {cluster_idx}")
            continue

        with open(caption_candidates[0], "r") as f:
            caption_data = json.load(f)

        caption = caption_data.get('caption', '')
        if not caption:
            print(f"Empty caption for cluster {cluster_idx}")
            continue

        prompt = f"{prompt_prefix} {caption}"
        print(f"Cluster {cluster_idx}: {prompt}")

        for synthesis_idx in range(n_images):
            out_path = f"{save_dir}/cluster_{cluster_idx:03d}_gemini_syn_{synthesis_idx:03d}.png"
            if os.path.exists(out_path):
                continue

            success = generate_image_with_gemini(prompt, out_path, client)
            if not success:
                print(f"    Failed to generate image {synthesis_idx} for cluster {cluster_idx}")


def evaluate_auc_mad(model_name, layer_name, neuron_idx, groups, save_dir):
    """Evaluate AUC and MAD for synthesized images."""
    safe_layer = layer_name.replace('.', '_')
    activation_dir = f'{PROJECT_DIR}/results/activations/imagenet'
    act_path = f'{activation_dir}/{model_name}/activations_{safe_layer}_output_raw.safetensors'

    act_actual = load_file(act_path)[layer_name]
    model_type = 'vit' if act_actual.ndim == 3 else 'conv'
    act_actual = aggregate_spatial_dimensions(act_actual, aggregation="top_mean", top_percentile=10.0, type=model_type)

    model = get_fn_model_loader(model_name)()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    extractor = ActivationExtractor(model, [(layer_name, "output")])

    non_concept_samples = np.random.randint(0, 50, size=1000) + np.arange(0, 50000, 50)
    act_non_concept = act_actual[non_concept_samples, neuron_idx]

    prompt_prefix = "realistic photo of a close up of"
    cluster_results = []

    for cluster_idx in trange(len(groups), desc="Evaluating"):
        sample_indices = groups[cluster_idx]
        if isinstance(sample_indices, np.ndarray):
            sample_indices = sample_indices.tolist()
        cluster_size = len(sample_indices)

        # Load caption
        caption_pattern = f'{save_dir}/gemini_caption_cluster_{cluster_idx:03d}_*.json'
        caption_candidates = glob.glob(caption_pattern)
        caption = None
        prompt = None
        if caption_candidates:
            with open(caption_candidates[0], "r") as f:
                caption_data = json.load(f)
            caption = caption_data.get('caption', '')
            prompt = f"{prompt_prefix} {caption}"

        syn_flist = sorted(glob.glob(f'{save_dir}/cluster_{cluster_idx:03d}_gemini_syn_*.png'))
        if not syn_flist:
            print(f"No synthetic images for cluster {cluster_idx}")
            cluster_results.append({
                'cluster_idx': cluster_idx,
                'cluster_size': cluster_size,
                'sample_indices': sample_indices,
                'caption': caption,
                'prompt': prompt,
                'n_synthetic_images': 0,
                'auc': None,
                'mad': None,
            })
            continue

        dataset = ImageDataset(syn_flist, transform=transform)
        data_loader = DataLoader(dataset, batch_size=50)

        result = extractor.extract(data_loader, save_dir=None, save_intermediate=False, pool_type="raw")
        act_simulated = result[(layer_name, 'output')]

        sim_type = 'vit' if act_simulated.ndim == 3 else 'conv'
        act_simulated = aggregate_spatial_dimensions(act_simulated, "top_mean", top_percentile=10.0, type=sim_type)
        act_concept = act_simulated[:, neuron_idx]

        concept_labels = torch.cat([torch.zeros(act_non_concept.shape[0]), torch.ones(act_concept.shape[0])], 0)
        act = torch.cat([act_non_concept, act_concept], 0)

        auc = roc_auc_score(concept_labels.cpu(), act.cpu())
        if act_non_concept.std().item() == 0:
            mad = 0.0
        else:
            mad = (act_concept.mean().item() - act_non_concept.mean().item()) / act_non_concept.std().item()

        print(f"Cluster {cluster_idx}: AUC={auc:.4f}, MAD={mad:.4f}")
        cluster_results.append({
            'cluster_idx': cluster_idx,
            'cluster_size': cluster_size,
            'sample_indices': sample_indices,
            'caption': caption,
            'prompt': prompt,
            'n_synthetic_images': len(syn_flist),
            'auc': auc,
            'mad': mad,
        })

    results = {
        'model_name': model_name,
        'layer_name': layer_name,
        'neuron_idx': neuron_idx,
        'num_clusters': len(groups),
        'non_concept_samples': non_concept_samples.tolist(),
        'clusters': cluster_results,
    }

    return results


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup Gemini
    api_key = args.gemini_api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Gemini API key required. Use --gemini_api_key or set GEMINI_API_KEY env var")

    client = genai.Client(api_key=api_key)

    model = get_fn_model_loader(args.model_name)()

    cluster_dir = f'{PROJECT_DIR}/results/clustering/kmeans_efficient'
    cluster_file = f'{cluster_dir}/{args.model_name}/{args.layer_name}/None/{args.neuron_idx:04d}/idx_00000_00100.pkl'
    with open(cluster_file, 'rb') as f:
        groups = pickle.load(f)

    output_dir = args.output_dir or f'{PROJECT_DIR}/results/caption/ours_gemini'
    save_dir = f'{output_dir}/{args.model_name}/{args.layer_name}/{args.neuron_idx:04d}'
    os.makedirs(save_dir, exist_ok=True)

    num_clusters = len(groups)

    if not args.skip_crop:
        for cluster_idx in trange(num_clusters, desc="Creating cluster grids"):
            sample_indices = groups[cluster_idx]
            save_path = f"{save_dir}/cluster_{cluster_idx:03d}_size_{len(sample_indices)}.png"
            if os.path.exists(save_path):
                continue
            create_cluster_grid(sample_indices, args.neuron_idx, model, args.layer_name, args.model_name, save_path)

    if not args.skip_caption:
        generate_captions(save_dir, client)

    if not args.skip_synthesis:
        synthesize_images(save_dir, num_clusters, args.n_images, client)

    results = evaluate_auc_mad(args.model_name, args.layer_name, args.neuron_idx, groups, save_dir)

    # Add additional metadata
    results['seed'] = args.seed
    results['n_images_per_cluster'] = args.n_images
    results['method'] = 'ours_gemini'

    results_path = os.path.join(save_dir, 'auc_mad_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print summary
    valid_clusters = [c for c in results['clusters'] if c['auc'] is not None]
    if valid_clusters:
        avg_auc = np.mean([c['auc'] for c in valid_clusters])
        avg_mad = np.mean([c['mad'] for c in valid_clusters])
        print(f"\nSummary: {len(valid_clusters)} clusters evaluated")
        print(f"Average AUC: {avg_auc:.4f}, Average MAD: {avg_mad:.4f}")


if __name__ == "__main__":
    main()

"""
python run_AUC_MAD_ours_gemini.py \
    --gemini_api_key AIzaSyAKxfcgYMtc2Mrbm4LO260JTa-2hBgOAHg \
    --model_name vit_b_16_timm \
    --layer_name blocks.6 \
    --neuron_idx 360 \
    --n_images 1
"""
