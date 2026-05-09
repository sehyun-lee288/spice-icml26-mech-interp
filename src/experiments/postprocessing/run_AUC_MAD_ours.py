import os
import argparse
import glob
import json
import re
import base64
import pickle
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from pydantic import create_model
from openai import OpenAI
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from models import get_fn_model_loader
from experiments.preprocessing.crop_activation_regions import get_cropped_images
from experiments.preprocessing.extract_activations import ActivationExtractor
from experiments.preprocessing.compute_top_activations import aggregate_spatial_dimensions
from experiments.disentangling.constants import PROJECT_DIR

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(42)

PROMPT_TEMPLATE = (
    'Given images, each containing highlighted regions, find some common objects and attributes in these images '
    'and describe each image with words especially repeated across these images.\n'
    'Your response should follow these rules: '
    '1. Pay more attention to the repeated objects or attributes across these images. '
    '2. Possible objects or attributes you can use to describe these images are '
    'object category, scene, object part, colour, texture, material, position,'
    'transparency, brightness, shape, size, edges, and their relationships. '
    '3. The identified common objects or attributes must appear simultaneously in at least 2 images. '
    '4. The identified specific objects or attributes represent some important contents of an individual image '
    'but not in the common objects or attributes found in the previous step. '
    '5. Your description of each image should be simple and only 3 words. '
    '6. Your response should be in the format of a JSON file, of which each key is a simple image index and '
    'each value is an object or attribute.\n'
    'Your identification process should strictly follow these steps: '
    'Step 1, take an overview of all images and summarize all possible common objects or attributes that appear simultaneously in at least any 2 of these images. '
    'Step 2, for each individual image, identify the common objects or attributes found in Step 1 that are also appear in the current image to describe the current image.'
    'Step 3, for each individual image, you can also use some specific attributes or objects that are not common across these images to describe the current image '
    'if there is not enough 3-word description for the common object or attribute found in Step 2.\n'
    'Now, please provide your response: '
)

SYSTEM_PROMPT = (
    "You are a helpful assistant designed to describe the commonality and "
    "specificity of the given images, and output a JSON format response."
)


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
    parser = argparse.ArgumentParser(description="Run AUC/MAD evaluation for our clustering method")
    parser.add_argument("--model_name", type=str, default="vit_b_16_timm")
    parser.add_argument("--layer_name", type=str, default="blocks.11")
    parser.add_argument("--neuron_idx", type=int, default=360)
    parser.add_argument("--n_images", type=int, default=3, help="Number of images to synthesize per cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--skip_crop", action="store_true")
    parser.add_argument("--skip_caption", action="store_true")
    parser.add_argument("--skip_synthesis", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_cluster_grid(sample_indices, neuron_idx, model, layer_name, model_name, save_path):
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


def generate_captions(save_dir, client):
    flist = sorted(glob.glob(f'{save_dir}/cluster_*_size_*.png'))

    for fname in flist:
        match = re.search(r'cluster_(\d+)', fname)
        cluster_idx = int(match.group(1)) if match else None

        match = re.search(r'size_(\d+)', fname)
        size = int(match.group(1)) if match else None

        out_fname = f'{save_dir}/concept_response_cluster_{cluster_idx:03d}_size_{size}.json'
        if os.path.exists(out_fname):
            continue

        base64_image = encode_image(fname)
        ConceptResponse = create_model(
            'ConceptResponse',
            step1_common=(list[str], ...),
            **{str(i): (str, ...) for i in range(size)}
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT_TEMPLATE},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        completion = client.chat.completions.parse(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            response_format=ConceptResponse,
        )

        resp = json.loads(completion.choices[0].message.content)
        with open(out_fname, 'w', encoding='utf-8') as f:
            json.dump(resp, f, indent=2, ensure_ascii=False)


def synthesize_images(save_dir, num_clusters, n_images, seed):
    prompt_prefix = "realistic photo of a close up of"

    for cluster_idx in tqdm(range(num_clusters), desc="Synthesizing images"):
        resp_pattern = f'{save_dir}/concept_response_cluster_{cluster_idx:03d}_*.json'
        resp_candidates = glob.glob(resp_pattern)
        if not resp_candidates:
            print(f"No response file matching pattern: {resp_pattern}")
            continue
        resp_fname = resp_candidates[0]

        with open(resp_fname, "r") as f:
            resp = json.load(f)

        prompt = f"{prompt_prefix} " + ', '.join(resp['step1_common'])
        print(prompt)

        for synthesis_idx in range(n_images):
            out_path = f"{save_dir}/cluster_{cluster_idx:03d}_syn_{synthesis_idx:03d}.png"
            if os.path.exists(out_path):
                continue
            image = pipe(prompt=prompt, generator=generator).images[0]
            image.save(out_path)


def evaluate_auc_mad(model_name, layer_name, neuron_idx, groups, save_dir):
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

        # Load prompt from response file
        resp_pattern = f'{save_dir}/concept_response_cluster_{cluster_idx:03d}_*.json'
        resp_candidates = glob.glob(resp_pattern)
        prompt = None
        concepts = None
        if resp_candidates:
            with open(resp_candidates[0], "r") as f:
                resp = json.load(f)
            concepts = resp.get('step1_common', [])
            prompt = f"{prompt_prefix} " + ', '.join(concepts)

        syn_flist = sorted(glob.glob(f'{save_dir}/cluster_{cluster_idx:03d}_syn_*.png'))
        if not syn_flist:
            print(f"No synthetic images for cluster {cluster_idx}")
            cluster_results.append({
                'cluster_idx': cluster_idx,
                'cluster_size': cluster_size,
                'sample_indices': sample_indices,
                'concepts': concepts,
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
            'concepts': concepts,
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

    model = get_fn_model_loader(args.model_name)()
    safe_layer = args.layer_name.replace('.', '_')

    cluster_dir = f'{PROJECT_DIR}/results/clustering/kmeans_efficient'
    cluster_file = f'{cluster_dir}/{args.model_name}/{args.layer_name}/None/{args.neuron_idx:04d}/idx_00000_00100.pkl'
    with open(cluster_file, 'rb') as f:
        groups = pickle.load(f)

    save_dir = f'{args.output_dir}/{args.model_name}/{args.layer_name}/{args.neuron_idx:04d}'
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
        api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Use --openai_api_key or set OPENAI_API_KEY env var")
        client = OpenAI(api_key=api_key)
        generate_captions(save_dir, client)

    if not args.skip_synthesis:
        synthesize_images(save_dir, num_clusters, args.n_images, args.seed)

    results = evaluate_auc_mad(args.model_name, args.layer_name, args.neuron_idx, groups, save_dir)

    # Add additional metadata
    results['seed'] = args.seed
    results['n_images_per_cluster'] = args.n_images

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
