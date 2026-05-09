import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from timm.data import create_transform, resolve_data_config

import os
import glob
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange 

from dsets import get_imagenet, get_dataset
from models import get_fn_model_loader, FEATURE_DIMS


# Basic logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Compute top activated sample indices from safetensors activation files")
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="vit_b_16_timm",
        help="Model name to process"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imagenet",
        help="Dataset name to process"
    )
    parser.add_argument(
        "--tgt_layer_name",
        type=str,
        default="blocks.11"
    )   
    parser.add_argument(
        "--src_layer_name",
        type=str,
        default="blocks.10",
    )
    parser.add_argument(
        "--neuron_indices", 
        type=int, 
        nargs='+',
        default=None,
        help="Which neurons to process (default: first 5 if --all_neurons not specified)"
    )
    parser.add_argument(
        "--all_neurons", 
        action="store_true",
        help="Process all available neurons (overrides --neuron_indices)"
    )
    parser.add_argument(
        "--top_index_file",
        type=str,
        default="/project/results/top_activations/top10pct/top_activations_blocks_11_output_indices.npy",
        help="A file storing highly activated sample indices"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Number of samples to compute attribution for top index file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/results/attributions/imagenet",
    )
    return parser.parse_args()



def get_layer_by_name(model, layer_name):
    parts = layer_name.split('.')
    # print (f"Getting layer by name: {layer_name} - {parts}")
    
    layer = model

    try:
        for i, part in enumerate(parts):
            if part.isdigit():
                # Handle numeric indices (e.g., blocks.0, layers.5)
                idx = int(part)
                if hasattr(layer, '__getitem__'):
                    layer = layer[idx]
                else:
                    # Try to get as attribute first (some models use numeric attributes)
                    try:
                        layer = getattr(layer, part)
                    except AttributeError:
                        print(f"Warning: Could not access index {idx} in layer at path: {'.'.join(parts[:i])}")
                        return None
            else:
                # Handle attribute access (e.g., encoder, blocks, layer1)
                if hasattr(layer, part):
                    layer = getattr(layer, part)
                else:
                    print(f"Warning: Attribute '{part}' not found in layer at path: {'.'.join(parts[:i])}")
                    return None
        return layer
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        print(f"Error accessing layer '{layer_name}': {e}")
        return None

class HookManager:
    """A context manager for safely managing PyTorch hooks."""
    def __init__(self):
        self.handles = []

    def register(self, handle):
        self.handles.append(handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()

    def remove_all(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

class AttributionExtractor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def compute_input_gradient(self,
                             input_tensor: torch.Tensor,
                             tgt_layer: nn.Module,
                             src_layer: nn.Module,
                             tgt_neurons: Optional[Union[int, slice, Sequence[int]]] = None
                             ) -> torch.Tensor:
        """
        Computes Input * Gradient for a specified source layer with respect to target neurons.
        This implementation is robust, memory-safe, and supports batch operations.

        Args:
            input_tensor: A batch of input tensors (e.g., shape [B, C, H, W])
            tgt_layer: The target layer module itself.
            src_layer: The source layer module itself.
                       The output activation of this module will be captured.
            tgt_neurons: Specifies the target neurons (channels) in the target layer.
                         If None, all neurons are used.

        Returns:
            A tensor containing the Input * Gradient attribution map.
        """
        self.model.eval()
        
        source_activations: Dict[str, torch.Tensor] = {}
        target_activation: Dict[str, torch.Tensor] = {}
        SRC_KEY = 'attr' # A constant key for our dictionaries

        with HookManager() as hook_manager:
            # --- Hook Definitions ---
            def forward_hook_fn(module: nn.Module, inp: Any, out: torch.Tensor):
                if not isinstance(out, torch.Tensor):
                    assert len(out) == 1, f"HARD CODE: out was expected to be len(out)=1, but {len(out)}"
                    out = out[0]
                source_activations[SRC_KEY] = out
                out.retain_grad()

            def target_forward_hook_fn(module: nn.Module, inp: Any, out: torch.Tensor):
                if not isinstance(out, torch.Tensor):
                    assert len(out) == 1, f"HARD CODE: out was expected to be len(out)=1, but {len(out)}"
                    out = out[0]
                target_activation['output'] = out

            # --- Register Hooks ---
            try:
                handle = src_layer.register_forward_hook(forward_hook_fn)
                hook_manager.register(handle)
            except Exception as e:
                logger.error(f"Source layer hook failed: {e}")
                raise

            try:
                handle = tgt_layer.register_forward_hook(target_forward_hook_fn)
                hook_manager.register(handle)
            except Exception as e:
                logger.error(f"Target layer hook failed: {e}")
                raise

            # --- Forward and Backward Pass ---
            input_tensor = input_tensor.to(self.device)
            self.model.zero_grad()
            self.model(input_tensor)

            if 'output' not in target_activation:
                raise RuntimeError(f"Failed to capture activation from target layer")

            target_act = target_activation['output']

            if tgt_neurons is not None:
                index = [slice(None)] * target_act.dim()
                
                # If the tensor is 3D (like ViT's B, seq_len, token_dim),
                # apply indexing to the last dimension (token_dim).
                # Otherwise, apply to the channel/feature dimension (dim 1).
                if target_act.dim() == 3:
                    dim_to_index = 2
                else:
                    dim_to_index = 1
                
                index[dim_to_index] = tgt_neurons
                selected_target = target_act[tuple(index)]
            else:
                selected_target = target_act

            target_scalar = selected_target.sum()
            target_scalar.backward()

        # --- Compute Attributions ---
        if SRC_KEY in source_activations and source_activations[SRC_KEY].grad is not None:
            act = source_activations[SRC_KEY]
            grad = source_activations[SRC_KEY].grad

            result = act * grad
            del act, grad  # To help prevent memory leak
            return result
        else:
            raise RuntimeError("Failed to capture activation or gradient for the source layer.")


def combine_attribution_files(attr_dir, model_name, layer_name):
    flist = sorted(glob.glob(os.path.join(attr_dir, 'attribution_*.safetensors')))

    # assert len(flist) == FEATURE_DIMS[model_name][layer_name], f"Files: {len(flist)}  should be {FEATURE_DIMS[model_name][layer_name]}"
    
    all_attr = None
    for i, fpath in tqdm(enumerate(flist), total=len(flist)):
        attr = load_file(fpath)['attribution']
        
        # Aggregation H, W or seq_len
        index = [slice(None)] * attr.dim()
        if attr.dim() == 3:
            attr = attr.sum(dim=1)
        elif attr.dim() == 4:
            attr = attr.sum(dim=(2,3))
        
        if all_attr is None:
            all_attr = torch.empty([len(flist), *attr.shape])
        else:
            all_attr[i] = attr
    
    # assert len(flist) == all_attr.size(0) == FEATURE_DIMS[model_name][layer_name], f"Files: {len(flist)}, all_attr: {all_attr.shape} should be {FEATURE_DIMS[model_name][layer_name]}"

    save_path = os.path.join(attr_dir, f'layer_attribution.safetensors')
    data_to_save = {'attribution': all_attr}
    save_file(data_to_save, save_path)


def main():
    args = get_args()

    # Load model and dataset
    model = get_fn_model_loader(args.model_name)()
    model.eval().cuda();

    
    # Setup data transforms
    if hasattr(model, 'pretrained_cfg'):
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        transform = transform.transforms
    else:
        transform = None
    
    # Load dataset
    dataset = get_dataset(args.dataset_name)(
        data_path=('/project/data/external/ILSVRC/Data/CLS-LOC' if args.dataset_name == 'imagenet' 
                   else '/project/dinov3/data' if args.dataset_name == 'food' 
                   else None),
        preprocessing=True,
        split="val",
        transform=transform,
    )
   
    # Load highly activated sample indices 
    top_indices = np.load(args.top_index_file)  # (channel OR token_dim, samples)

    # Set (input * gradient) extractor
    device = 'cuda'
    extractor = AttributionExtractor(model, device) 
    tgt_layer_module = get_layer_by_name(model, args.tgt_layer_name)
    src_layer_module = get_layer_by_name(model, args.src_layer_name)
   
    
    # Set neurons
    if args.all_neurons:
        neuron_indices_to_process = list(range(top_indices.shape[0]))
        print(f"Processing all {len(neuron_indices_to_process)} neurons")
    elif args.neuron_indices is not None:
        # Process specified neurons
        neuron_indices_to_process = args.neuron_indices
        print(f"Processing {len(neuron_indices_to_process)} specified neurons: {neuron_indices_to_process}")
    else:
        # Default: process first 5 neurons
        neuron_indices_to_process = [0, 1, 2, 3, 4]
        print(f"Processing default neurons: {neuron_indices_to_process}")


    save_dir = os.path.join(args.output_dir, args.model_name, args.tgt_layer_name)
    os.makedirs(save_dir, exist_ok=True)

    for target_neurons_slice in tqdm(neuron_indices_to_process):
        subset = torch.utils.data.Subset(dataset, top_indices[target_neurons_slice][:args.max_samples])
        dataloader = torch.utils.data.DataLoader(subset, batch_size=128,
                                                 shuffle=False, num_workers=8, pin_memory=True)

        all_attributions = []

        for image, _ in dataloader:
            input_batch = image.cuda()

            attribution_map = extractor.compute_input_gradient(
                                  input_tensor=input_batch,
                                  tgt_layer=tgt_layer_module,
                                  src_layer=src_layer_module,
                                  tgt_neurons=target_neurons_slice)

            if attribution_map.dim() == 3: # (samples, seq_len, token_dim)
                attribution_map = attribution_map.sum(dim=1)
            elif attribution_map.dim() == 4: # (samples, C, H, W)
                attribution_map = attribution_map.sum(dim=(2,3))

            all_attributions.append(attribution_map.detach().cpu())

            # Free memory of intermediate tensors to avoid memory leak
            del attribution_map
            del input_batch

        all_attributions = torch.cat(all_attributions, dim=0)

        # Save attribution_map for all subset
        save_path = os.path.join(save_dir, f'attribution_{target_neurons_slice:04d}.safetensors')
        data_to_save = {'attribution': all_attributions}
        save_file(data_to_save, save_path)

        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
