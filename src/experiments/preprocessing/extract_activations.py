import argparse
import os
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from safetensors.torch import save_file

from dsets import get_dataset
from models import get_fn_model_loader
from utils.helper import load_config, get_layer_names_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ActivationExtractor:
    """Extract activations from multiple layers of a model"""
    
    def __init__(self, model: nn.Module, layers_to_hook: List[str]):
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.hooks = {}
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for specified layers"""
        def get_activation(name, target_type='output'): # 기본값은 'output'
            def hook(model, input, output):
                tensor_to_save = None
                
                # target_type에 따라 저장할 텐서를 결정
                if target_type == 'output':
                    # 기존 output 저장 로직
                    if isinstance(output, torch.Tensor):
                        tensor_to_save = output
                    elif isinstance(output, (tuple, list)):
                        tensor_to_save = output[0]
                
                elif target_type == 'input':
                    # input 저장 로직 (input은 항상 튜플이므로 첫 번째 요소 사용)
                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        tensor_to_save = input[0]

                # 텐서가 정상적으로 선택되었으면 저장
                if tensor_to_save is not None:
                    self.activations[name] = tensor_to_save.detach().cpu()

            return hook
        
        # Register hooks for each layer
        for layer_name, target_type in self.layers_to_hook: # self.layers_to_hook로 변경
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                # get_activation에 target_type도 함께 전달
                handle = layer.register_forward_hook(get_activation(layer_name, target_type))
                # hook 핸들을 저장할 때도 (이름, 타입)으로 저장하면 더 명확할 수 있습니다.
                self.hooks[(layer_name, target_type)] = handle 
            else:
                print(f"Warning: Layer '{layer_name}' not found in model")
                raise ValueError(f"Layer '{layer_name}' not found in model")
        

    def _get_layer_by_name(self, layer_name: str, model=None):
        """Get layer object by name, supporting nested paths like 'encoder.blocks.0'"""
        parts = layer_name.split('.')
        print (f"Getting layer by name: {layer_name} - {parts}")
        if self.model is None and model is not None:
            layer = model
        layer = self.model
        
        
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
    
    def extract(self, data_loader: DataLoader, save_dir: str = None, 
                save_intermediate: bool = False, pool_type: str = "raw", 
                checkpoint_interval: int = 100) -> Dict[str, torch.Tensor]:
        """Extract activations for all samples in the data loader
        
        When save_intermediate=True, activations are saved to checkpoint files
        every checkpoint_interval batches and immediately deleted from memory
        to prevent memory overflow. After processing all samples, checkpoint
        files are combined layer by layer into final safetensors files.
        """
        all_activations = {(name, target_type): [] for name, target_type in self.layers_to_hook}
        
        self.model.eval()
        total_batches = len(data_loader)
        checkpoint_counter = 0  # Track checkpoint numbers
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Extracting activations")):
                inputs = inputs.to(next(self.model.parameters()).device)
                
                # Clear previous activations
                self.activations.clear()
                
                # Forward pass (triggers hooks)
                _ = self.model(inputs)
                
                # Collect activations
                for j, (layer_name, target_type) in enumerate(self.layers_to_hook):
                    if layer_name in self.activations:
                        if j==0: print (f"Layer {layer_name} has {self.activations[layer_name].shape} activations")
                        all_activations[(layer_name, target_type)].append(self.activations[layer_name])
                
                # Save intermediate results if requested and at checkpoint interval
                if save_intermediate and save_dir and (batch_idx + 1) % checkpoint_interval == 0:
                    checkpoint_counter += 1
                    self._save_intermediate_checkpoint(all_activations, save_dir, checkpoint_counter, pool_type)
                    # Clear activations from memory after saving to prevent memory overflow
                    for key in all_activations:
                        all_activations[key].clear()
        
        # Handle final batch if save_intermediate is True and there are remaining activations
        if save_intermediate and save_dir and any(all_activations.values()):
            checkpoint_counter += 1
            self._save_intermediate_checkpoint(all_activations, save_dir, checkpoint_counter, pool_type)
            # Clear final activations
            for key in all_activations:
                all_activations[key].clear()
        
        # Process and save final results
        if save_dir:
            if save_intermediate:
                # Combine checkpoint files layer by layer
                self._combine_checkpoint_files(save_dir, checkpoint_counter, pool_type)
            else:
                # Use traditional method for non-intermediate saving
                self._save_layer_by_layer(all_activations, save_dir, pool_type)
            return {}  # Return empty dict since files are saved
        else:
            # Concatenate all batches for return (only when not saving to disk)
            result = {}
            for layer_name, target_type in self.layers_to_hook:
                if all_activations[(layer_name, target_type)]:
                    result[(layer_name, target_type)] = torch.cat(all_activations[(layer_name, target_type)], dim=0)
                else:
                    print(f"Warning: No activations collected for layer '{layer_name}'")
            return result
    
    def _save_intermediate_checkpoint(self, all_activations: Dict[str, List], 
                                     save_dir: str, checkpoint_num: int, pool_type: str):
        """Save intermediate checkpoint"""
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                # Concatenate current activations
                activations = torch.cat(all_activations[(layer_name, target_type)], dim=0)
                
                # Save checkpoint without pooling (pooling will be applied during combination)
                checkpoint_path = os.path.join(checkpoint_dir, f"{layer_name}_batch_{checkpoint_num}.pt")
                torch.save(activations, checkpoint_path)
                print(f"Saved checkpoint {checkpoint_num} for {layer_name} with shape {activations.shape}")
                
                # Free memory immediately after saving
                del activations
    
    def _apply_pooling(self, activation: torch.Tensor, pool_type: str) -> torch.Tensor:
        """Apply pooling to activation tensor"""
        if activation.dim() <= 2:
            return activation
        
        if pool_type == "gap":  # Global Average Pooling
            return activation.mean(dim=list(range(2, activation.dim())))
        elif pool_type == "gmp":  # Global Max Pooling
            for dim in range(activation.dim() - 1, 1, -1):
                activation = activation.max(dim=dim)[0]
            return activation
        elif pool_type == "raw":  # No pooling
            return activation
        elif pool_type == "top_mean":
            from experiments.preprocessing.compute_top_activations import aggregate_spatial_dimensions
            model_type = 'vit' if activation.ndim == 3 else 'conv'
            return aggregate_spatial_dimensions(activation, "top_mean", top_percentile=10.0, type=model_type)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
    
    def _save_layer_by_layer(self, all_activations: Dict[str, List], save_dir: str, pool_type: str):
        """Save activations layer by layer"""
        os.makedirs(save_dir, exist_ok=True)
        
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                print(f"Processing and saving layer: {layer_name}")
                
                # Concatenate all batches for this layer
                activations = torch.cat(all_activations[(layer_name, target_type)], dim=0)
                
                # Apply pooling
                activations = self._apply_pooling(activations, pool_type)
                
                print(f"  Final shape: {activations.shape}")
                
                # Create safe filename from layer name
                safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                filename = f"activations_{safe_layer_name}_{target_type}_{pool_type}.safetensors"
                save_path = os.path.join(save_dir, filename)
                
                # Save individual layer
                save_file({layer_name: activations.contiguous()}, save_path)
                print(f"  Saved to: {save_path}")
                
                # Save metadata for each layer
                metadata = {
                    "layer_name": layer_name,
                    "shape": list(activations.shape),
                    "pool_type": pool_type,
                    "dtype": str(activations.dtype)
                }
                
                metadata_path = save_path.replace(".safetensors", "_metadata.txt")
                with open(metadata_path, "w") as f:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                
                # Free memory
                del activations
                all_activations[(layer_name, target_type)].clear()
                
            else:
                print(f"Warning: No activations collected for layer '{layer_name}'")
    
    def _combine_checkpoint_files(self, save_dir: str, checkpoint_counter: int, pool_type: str):
        """Combine checkpoint files for each layer into a single safetensors file."""
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for layer_name, target_type in self.layers_to_hook:
            combined_activations = []
            for i in range(1, checkpoint_counter + 1):
                checkpoint_path = os.path.join(checkpoint_dir, f"{layer_name}_batch_{i}.pt")
                if os.path.exists(checkpoint_path):
                    try:
                        activations = torch.load(checkpoint_path)
                        # Apply pooling if needed
                        activations = self._apply_pooling(activations, pool_type)
                        combined_activations.append(activations)
                    except Exception as e:
                        print(f"Error loading or processing checkpoint {checkpoint_path}: {e}")
                        continue

            if combined_activations:
                combined_activations = torch.cat(combined_activations, dim=0)
                print(f"Combining {checkpoint_counter} checkpoints for {layer_name}")
                
                # Create safe filename from layer name
                safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                filename = f"activations_{safe_layer_name}_{target_type}_{pool_type}.safetensors"
                save_path = os.path.join(save_dir, filename)
                
                # Save combined activations
                save_file({layer_name: combined_activations}, save_path)
                print(f"  Saved combined to: {save_path}")
                
                # Save metadata for combined layer
                metadata = {
                    "layer_name": layer_name,
                    "shape": list(combined_activations.shape),
                    "pool_type": pool_type,
                    "dtype": str(combined_activations.dtype),
                    "num_checkpoints": checkpoint_counter
                }
                metadata_path = save_path.replace(".safetensors", "_metadata.txt")
                with open(metadata_path, "w") as f:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                
                # Free memory
                del combined_activations
                
                # Clean up intermediate checkpoint files for this layer
                for i in range(1, checkpoint_counter + 1):
                    checkpoint_path = os.path.join(checkpoint_dir, f"{layer_name}_batch_{i}.pt")
                    if os.path.exists(checkpoint_path):
                        try:
                            os.remove(checkpoint_path)
                            print(f"  Removed checkpoint file: {checkpoint_path}")
                        except OSError as e:
                            print(f"  Warning: Could not remove checkpoint file {checkpoint_path}: {e}")
            else:
                print(f"Warning: No valid checkpoint files found for layer '{layer_name}' to combine.")
        
        # Remove checkpoint directory if it's empty
        try:
            if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)
                print(f"Removed empty checkpoint directory: {checkpoint_dir}")
        except OSError as e:
            print(f"Warning: Could not remove checkpoint directory {checkpoint_dir}: {e}")
    
    def print_model_structure(self, max_depth=3, prefix=""):
        """Print the model structure to help identify layer names"""
        def _print_structure(module, current_depth=0, prefix=""):
            if current_depth > max_depth:
                return
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                print(f"{'  ' * current_depth}{full_name}: {type(child).__name__}")
                
                # Print some key attributes for common layer types
                if hasattr(child, 'num_features'):
                    print(f"{'  ' * current_depth}  └─ num_features: {child.num_features}")
                elif hasattr(child, 'out_features'):
                    print(f"{'  ' * current_depth}  └─ out_features: {child.out_features}")
                elif hasattr(child, 'num_channels'):
                    print(f"{'  ' * current_depth}  └─ num_channels: {child.num_channels}")
                
                # Recursively print children
                _print_structure(child, current_depth + 1, full_name)
        
        print(f"Model structure (max depth: {max_depth}):")
        _print_structure(self.model, 0, "")
    
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


def get_args():
    parser = argparse.ArgumentParser(description="Extract activations from specified layers")
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="configs/imagenet/vit_b_16_timm.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Model name (overrides config file if provided)"
    )
    parser.add_argument(
        "--layers_to_hook",
        type=str,
        nargs='+',
        default=['blocks.0.attn', 'output', 'blocks.11.norm2', 'input'], 
        help="Pairs of layer_name and target_type ('input' or 'output')"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/activations/imagenet",
        help="Directory to save activation results"
    )
    parser.add_argument(
        "--pool_type", 
        type=str, 
        default="raw", 
        choices=["gap", "gmp", "raw", "top_mean"],
        help="Pooling type: gap (global average), gmp (global max), raw (no pooling)"
    )
    parser.add_argument(
        "--print_model_structure",
        action="store_true",
        help="Print model structure and exit (useful for finding layer names)"
    )
    parser.add_argument(
        "--structure_depth",
        type=int,
        default=3,
        help="Maximum depth for printing model structure"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate results to checkpoint files and delete from memory to prevent memory overflow. Files are combined after processing all samples."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing intermediate results"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save checkpoint every N batches (only when save_intermediate is True)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="ckpt_path is specified either config.yaml or here or just use default weight in models directory"
    )
    
    args = parser.parse_args()
    
    flat_list = args.layers_to_hook
    if len(flat_list) % 2 != 0:
        raise ValueError("--layers_to_hook 인자는 반드시 '이름 타입' 쌍으로 주어져야 합니다. 예시: --layers_to_hook blocks.0.attn output blocks.11.norm2 input")
      
    args.layers_to_hook = list(zip(flat_list[::2], flat_list[1::2]))
    
    return args





def main():
    args = get_args()
    print(f"args.layers_to_hook: {args.layers_to_hook}")
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Use command line model_name if provided, otherwise use config
    model_name = args.model_name if args.model_name else config["model_name"]
    dataset_name = config["dataset_name"]
    data_path = config.get("data_path", None)
    ckpt_path = config.get("ckpt_path", None)
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if dataset_name == 'imagenet':
        num_classes = 1000
    else:
        # Try to infer from config or use default
        num_classes = config.get("num_classes", 1000)
        
    model = get_fn_model_loader(model_name)(
        ckpt_path=ckpt_path, n_class=num_classes
    ).to(device).eval()
    print(f"Model '{model_name}' loaded")
    
    # If user wants to print model structure, do it and exit
    if args.print_model_structure:
        extractor = ActivationExtractor(model, [])  # Empty layer list for structure printing
        extractor.print_model_structure(max_depth=args.structure_depth)
        return
    
    # Setup data transforms
    if hasattr(model, 'pretrained_cfg'):
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        transform = transform.transforms
    else:
        transform = None
    
    # Load dataset
    dataset = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=True,
        split="val",
        transform=transform,
    )
    print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples")
    
    # Limit samples if specified
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    if not args.layers_to_hook:
        print("Error: No valid layer names provided")
        return
    
    print(f"Extracting activations from layers: {args.layers_to_hook}")
    
    # Set up save directory
    save_dir = os.path.join(
        args.save_dir, 
        dataset_name, 
        model_name
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract activations
    extractor = ActivationExtractor(model, args.layers_to_hook)
    
    try:
        # Extract and save layer by layer
        activations = extractor.extract(
            data_loader, 
            save_dir=save_dir, 
            save_intermediate=args.save_intermediate,
            pool_type=args.pool_type,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save overall metadata
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "layers_to_hook": args.layers_to_hook,
            "num_samples": len(dataset),
            "pool_type": args.pool_type,
            "batch_size": args.batch_size,
            "total_layers": len(args.layers_to_hook),
        }
        
        metadata_path = os.path.join(save_dir, "extraction_metadata.txt")
        with open(metadata_path, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nExtraction complete!")
        print(f"Results saved to: {save_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        # List saved files
        print("\nSaved files:")
        for file in os.listdir(save_dir):
            if file.endswith('.safetensors'):
                file_path = os.path.join(save_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  {file} ({file_size:.2f} MB)")
        
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main() 
    
"""
python src/experiments/preprocessing/extract_activations.py \
    --layers_to_hook blocks.6 output \
    --pool_type raw \
    --model_name vit_b_16_timm \
    --config_file src/configs/imagenet/vit_b_16_timm.yaml \
    --save_dir results/activations
    
    
python src/experiments/preprocessing/extract_activations.py \
    --layers_to_hook blocks.10 output \
                     blocks.11.norm1 output \
                     blocks.11.attn output \
                     blocks.11.ls1 output \
                     blocks.11.drop_path1 output \
                     blocks.11.norm2 input \
                     blocks.11.norm2 output \
                     blocks.11.mlp.fc1 output \
                     blocks.11.mlp.act output \
                     blocks.11.mlp.fc2 output \
                     blocks.11.ls2 output \
                     blocks.11.drop_path2 output \
                     blocks.11 output \
    --pool_type raw \
    --model_name vit_b_16_timm \
    --config_file src/configs/imagenet/vit_b_16_timm.yaml \
    --save_dir results/activations/imagenet \
    --save_intermediate 

# Print model structure to find layer names
python src/experiments/preprocessing/extract_activations.py \
    --config_file src/configs/imagenet/vit_b_16_timm.yaml \
    --print_model_structure \
    --structure_depth 4

# Extract activations from nested ViT layers (saves each layer separately)
python src/experiments/preprocessing/extract_activations.py \
    --layers_to_hook blocks.0 output blocks.5 output blocks.11 output \
    --pool_type raw \
    --model_name vit_b_16_timm \
    --config_file src/configs/imagenet/vit_b_16_timm.yaml \
    --save_dir results/activations/imagenet

# Extract with intermediate checkpoints (saves every 100 batches)
python src/experiments/preprocessing/extract_activations.py \
    --layers_to_hook blocks.11 output \
    --pool_type raw \
    --model_name vit_b_16_timm \
    --config_file src/configs/imagenet/vit_b_16_timm.yaml \
    --save_dir results/activations/imagenet \
    --save_intermediate

# Extract from ResNet layers (saves each layer separately)
python src/experiments/preprocessing/extract_activations.py \
    --layers_to_hook layer1.0 output layer2.0 output layer3.0 output layer4.0 output \
    --pool_type gap \
    --model_name resnet50_timm \
    --config_file src/configs/imagenet/resnet50_timm.yaml \
    --save_dir results/activations/imagenet
    
"""