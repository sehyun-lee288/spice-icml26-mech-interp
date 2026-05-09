import argparse
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import json


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
        "--input_file", 
        type=str, 
        required=False,
        default=None,
        help="Path to safetensors activation file"
    )
    parser.add_argument(
        "--layer_name", 
        type=str, 
        required=False,
        default=None,
        help="Layer name to process"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/top_activations",
        help="Directory to save results"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=100,
        help="Number of top activated samples to find for each neuron/channel"
    )
    parser.add_argument(
        "--aggregation", 
        type=str, 
        default="max",
        choices=["max", "mean", "sum", "top_mean"],
        help="How to aggregate spatial dimensions (for conv layers)"
    )
    parser.add_argument(
        "--top_percentile", 
        type=float, 
        default=10.0,
        help="Percentage of top pixels to use for top_mean aggregation (default: 10.0)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1000,
        help="Batch size for processing (to manage memory)"
    )
    parser.add_argument(
        "--save_values", 
        action="store_true",
        help="Save activation values along with indices"
    )
    parser.add_argument(
        "--percentile", 
        type=float, 
        default=None,
        help="Only consider samples above this percentile (0-100)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="top",
        choices=["top", "middle", "bottom"],
        help="top-k, middle-k, bottom-k"
    )
    return parser.parse_args()


def load_activation_file(file_path: str) -> Dict[str, torch.Tensor]:
    """Load activations from safetensors file"""
    print(f"Loading activation file: {file_path}")
    
    activations = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        print(f"Available keys: {list(f.keys())}")
        for key in f.keys():
            tensor = f.get_tensor(key)
            activations[key] = tensor
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    
    return activations


def aggregate_spatial_dimensions(tensor: torch.Tensor, aggregation: str, top_percentile: float = 10.0, type='vit') -> torch.Tensor:
    """
    Aggregate spatial dimensions of activation tensor
    
    Args:
        tensor: Input tensor with shape (batch_size, channels, height, width, ...) or (batch_size, seq_len, hidden_dim)
        aggregation: Aggregation method ('max', 'mean', 'sum', 'top_mean')
        top_percentile: For 'top_mean', percentage of top pixels to average (0-100)
    
    Returns:
        Aggregated tensor with shape (batch_size, channels) or (batch_size, hidden_dim)
    """
    if tensor.dim() <= 2:
        raise ValueError(f"Expect tensor to have at least 3 dimensions, got {tensor.dim()}. If you are using a single neuron, do unsqueeze(2) first.")
    
    # Determine the tensor format
    if tensor.dim() == 3:
        # Could be (batch_size, seq_len, hidden_dim) for ViT or (batch_size, channels, spatial) for conv
        batch_size, dim1, dim2 = tensor.shape
        if type == 'vit':
            print(f"Detected ViT format: ({batch_size}, {dim1}, {dim2}) - aggregating over seq_len")
            return aggregate_vit_sequence(tensor, aggregation, top_percentile)
        elif type == 'conv':
            print(f"Detected conv format: ({batch_size}, {dim1}, {dim2}) - aggregating over spatial")
            return aggregate_conv_spatial(tensor, aggregation, top_percentile)
        else:
            raise ValueError(f"Unknown type: {type}")
    else:
        # 4D+ tensor, assume conv format: (batch_size, channels, height, width, ...)
        print(f"Detected conv format: {tensor.shape} - aggregating over spatial dimensions")
        return aggregate_conv_spatial(tensor, aggregation, top_percentile)


def aggregate_vit_sequence(tensor: torch.Tensor, aggregation: str, top_percentile: float = 10.0) -> torch.Tensor:
    """Aggregate over sequence dimension for ViT activations (batch_size, seq_len, hidden_dim)"""
    if aggregation == "max":
        return tensor.max(dim=1)[0]  # (batch_size, hidden_dim)
    elif aggregation == "mean":
        return tensor.mean(dim=1)  # (batch_size, hidden_dim)
    elif aggregation == "sum":
        return tensor.sum(dim=1)  # (batch_size, hidden_dim)
    elif aggregation == "top_mean":
        batch_size, seq_len, hidden_dim = tensor.shape
        # For each sample and each hidden dimension, take mean of top percentile sequence positions
        k = max(1, int(seq_len * top_percentile / 100.0))
        top_values, _ = torch.topk(tensor, k=k, dim=1, largest=True)
        return top_values.mean(dim=1)  # (batch_size, hidden_dim)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def aggregate_conv_spatial(tensor: torch.Tensor, aggregation: str, top_percentile: float = 10.0) -> torch.Tensor:
    """Aggregate over spatial dimensions for conv activations (batch_size, channels, height, width, ...)"""
    if aggregation == "max":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.max(dim=dim)[0]
    elif aggregation == "mean":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.mean(dim=dim)
    elif aggregation == "sum":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.sum(dim=dim)
    elif aggregation == "top_mean":
        # Calculate mean of top percentile pixels
        # Flatten spatial dimensions while keeping batch and channel dims
        spatial_dims = list(range(2, tensor.dim()))
        
        if len(spatial_dims) > 0:
            # Reshape to (batch_size, channels, spatial_pixels)
            batch_size, channels = tensor.shape[:2]
            spatial_size = 1
            for dim in spatial_dims:
                spatial_size *= tensor.shape[dim]
            
            tensor = tensor.view(batch_size, channels, spatial_size)
            
            # Calculate number of top pixels to keep
            k = max(1, int(spatial_size * top_percentile / 100.0))
            
            # Get top-k values across spatial dimension and take their mean
            top_values, _ = torch.topk(tensor, k=k, dim=2, largest=True)
            tensor = top_values.mean(dim=2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
        
    return tensor


def compute_top_activations(
    activations: torch.Tensor, 
    top_k: int, 
    aggregation: str = "max",
    percentile: Optional[float] = None,
    batch_size: int = 1000,
    device: str = "cpu",
    top_percentile: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k activated samples for each neuron/channel
    
    Args:
        activations: Tensor of shape (num_samples, num_neurons, ...)
        top_k: Number of top samples to return
        aggregation: How to aggregate spatial dimensions ('max', 'mean', 'sum', 'top_mean')
        percentile: Only consider samples above this percentile
        batch_size: Batch size for processing
        device: Device to use
        top_percentile: For 'top_mean' aggregation, percentage of top pixels to average
    
    Returns:
        top_indices: Tensor of shape (num_neurons, top_k) with sample indices
        top_values: Tensor of shape (num_neurons, top_k) with activation values
    """
    print(f"Computing top-{top_k} activations...")
    print(f"Input shape: {activations.shape}")
    
    # Aggregate spatial dimensions
    if activations.dim() > 2:
        if aggregation == "top_mean":
            print(f"Aggregating spatial dimensions using {aggregation} (top {top_percentile}% pixels)")
        else:
            print(f"Aggregating spatial dimensions using {aggregation}")
        aggregated = aggregate_spatial_dimensions(activations, aggregation, top_percentile)
        print(f"Aggregated shape: {aggregated.shape}")
    else:
        aggregated = activations
    
    num_samples, num_neurons = aggregated.shape
    print(f"Processing {num_samples} samples, {num_neurons} neurons")
    
    # Apply percentile filtering if specified
    if percentile is not None:
        print(f"Filtering samples above {percentile}th percentile")
        percentile_threshold = torch.quantile(aggregated, percentile / 100.0, dim=0)
        mask = aggregated >= percentile_threshold.unsqueeze(0)
        # Set values below threshold to -inf so they won't be selected
        aggregated = aggregated.clone()
        aggregated[~mask] = float('-inf')
    
    # Initialize result tensors
    top_indices = torch.zeros((num_neurons, top_k), dtype=torch.long)
    top_values = torch.zeros((num_neurons, top_k), dtype=aggregated.dtype)
    
    # Process neurons in batches to manage memory
    for start_idx in tqdm(range(0, num_neurons, batch_size), desc="Processing neurons"):
        end_idx = min(start_idx + batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(device)
        
        # Get top-k for this batch
        batch_top_values, batch_top_indices = torch.topk(
            batch_activations, 
            k=min(top_k, num_samples), 
            dim=0, 
            largest=True
        )
        
        # Store results
        actual_k = batch_top_indices.shape[0]
        top_indices[start_idx:end_idx, :actual_k] = batch_top_indices.T.cpu()
        top_values[start_idx:end_idx, :actual_k] = batch_top_values.T.cpu()
        
        # If we have fewer samples than top_k, pad with -1 for indices
        if actual_k < top_k:
            top_indices[start_idx:end_idx, actual_k:] = -1
            top_values[start_idx:end_idx, actual_k:] = float('-inf')
    
    return top_indices, top_values

def compute_middle_activations(
    activations: torch.Tensor, 
    top_k: int, 
    aggregation: str = "max",
    percentile: Optional[float] = None,
    batch_size: int = 1000,
    device: str = "cpu",
    top_percentile: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k activated samples for each neuron/channel
    
    Args:
        activations: Tensor of shape (num_samples, num_neurons, ...)
        top_k: Number of top samples to return
        aggregation: How to aggregate spatial dimensions ('max', 'mean', 'sum', 'top_mean')
        percentile: Only consider samples above this percentile
        batch_size: Batch size for processing
        device: Device to use
        top_percentile: For 'top_mean' aggregation, percentage of top pixels to average
    
    Returns:
        top_indices: Tensor of shape (num_neurons, top_k) with sample indices
        top_values: Tensor of shape (num_neurons, top_k) with activation values
    """
    print(f"Computing top-{top_k} activations...")
    print(f"Input shape: {activations.shape}")
    
    # Aggregate spatial dimensions
    if activations.dim() > 2:
        if aggregation == "top_mean":
            print(f"Aggregating spatial dimensions using {aggregation} (top {top_percentile}% pixels)")
        else:
            print(f"Aggregating spatial dimensions using {aggregation}")
        aggregated = aggregate_spatial_dimensions(activations, aggregation, top_percentile)
        print(f"Aggregated shape: {aggregated.shape}")
    else:
        aggregated = activations
    
    num_samples, num_neurons = aggregated.shape
    print(f"Processing {num_samples} samples, {num_neurons} neurons")
    
    # Apply percentile filtering if specified
    if percentile is not None:
        print(f"Filtering samples above {percentile}th percentile")
        percentile_threshold = torch.quantile(aggregated, percentile / 100.0, dim=0)
        mask = aggregated >= percentile_threshold.unsqueeze(0)
        # Set values below threshold to -inf so they won't be selected
        aggregated = aggregated.clone()
        aggregated[~mask] = float('-inf')
    
    # Initialize result tensors
    top_indices = torch.zeros((num_neurons, top_k), dtype=torch.long)
    top_values = torch.zeros((num_neurons, top_k), dtype=aggregated.dtype)

    # Process neurons in batches to manage memory
    for start_idx in tqdm(range(0, num_neurons, batch_size), desc="Processing neurons"):
        end_idx = min(start_idx + batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(device)

        # Sort activations for each neuron in ascending order
        sorted_values, sorted_indices = torch.sort(batch_activations, dim=0, descending=False)
        num_samples_batch = sorted_values.shape[0]

        # Select the "middle" k values/indices:
        # If num_samples_batch >= top_k, select k values centered around the median
        if num_samples_batch >= top_k:
            mid = num_samples_batch // 2
            half_k = top_k // 2
            # For even k we'll take k//2 below and above median
            if top_k % 2 == 0:
                start_mid = max(0, mid - half_k)
                end_mid = start_mid + top_k
            else:
                start_mid = max(0, mid - half_k)
                end_mid = start_mid + top_k

            # Clamp in case at the border
            if end_mid > num_samples_batch:
                end_mid = num_samples_batch
                start_mid = end_mid - top_k
            middle_indices = slice(start_mid, end_mid)

            batch_middle_indices = sorted_indices[middle_indices, :]
            batch_middle_values = sorted_values[middle_indices, :]
        else:
            # If there are not enough samples to take middle
            batch_middle_indices = sorted_indices[:, :]
            batch_middle_values = sorted_values[:, :]

        actual_k = batch_middle_indices.shape[0]
        top_indices[start_idx:end_idx, :actual_k] = batch_middle_indices.T.cpu()
        top_values[start_idx:end_idx, :actual_k] = batch_middle_values.T.cpu()

        # If we have fewer samples than top_k, pad with -1 for indices and -inf for values
        if actual_k < top_k:
            top_indices[start_idx:end_idx, actual_k:] = -1
            top_values[start_idx:end_idx, actual_k:] = float('-inf')

    return top_indices, top_values


def compute_bottom_activations(
    activations: torch.Tensor, 
    top_k: int, 
    aggregation: str = "max",
    percentile: Optional[float] = None,
    batch_size: int = 1000,
    device: str = "cpu",
    top_percentile: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k activated samples for each neuron/channel
    
    Args:
        activations: Tensor of shape (num_samples, num_neurons, ...)
        top_k: Number of top samples to return
        aggregation: How to aggregate spatial dimensions ('max', 'mean', 'sum', 'top_mean')
        percentile: Only consider samples above this percentile
        batch_size: Batch size for processing
        device: Device to use
        top_percentile: For 'top_mean' aggregation, percentage of top pixels to average
    
    Returns:
        top_indices: Tensor of shape (num_neurons, top_k) with sample indices
        top_values: Tensor of shape (num_neurons, top_k) with activation values
    """
    print(f"Computing bottom-{top_k} activations...")
    print(f"Input shape: {activations.shape}")
    
    # Aggregate spatial dimensions
    if activations.dim() > 2:
        if aggregation == "top_mean":
            print(f"Aggregating spatial dimensions using {aggregation} (top {top_percentile}% pixels)")
        else:
            print(f"Aggregating spatial dimensions using {aggregation}")
        aggregated = aggregate_spatial_dimensions(activations, aggregation, top_percentile)
        print(f"Aggregated shape: {aggregated.shape}")
    else:
        aggregated = activations
    
    num_samples, num_neurons = aggregated.shape
    print(f"Processing {num_samples} samples, {num_neurons} neurons")
    
    # Apply percentile filtering if specified
    if percentile is not None:
        print(f"Filtering samples above {percentile}th percentile")
        percentile_threshold = torch.quantile(aggregated, percentile / 100.0, dim=0)
        mask = aggregated >= percentile_threshold.unsqueeze(0)
        # Set values below threshold to -inf so they won't be selected
        aggregated = aggregated.clone()
        aggregated[~mask] = float('-inf')
    
    # Initialize result tensors
    bottom_indices = torch.zeros((num_neurons, top_k), dtype=torch.long)
    bottom_values = torch.zeros((num_neurons, top_k), dtype=aggregated.dtype)
    
    # Process neurons in batches to manage memory
    for start_idx in tqdm(range(0, num_neurons, batch_size), desc="Processing neurons"):
        end_idx = min(start_idx + batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(device)
        
        # Get bottom-k for this batch
        batch_bottom_values, batch_bottom_indices = torch.topk(
            batch_activations, 
            k=min(top_k, num_samples), 
            dim=0, 
            largest=False
        )
        
        # Store results
        actual_k = batch_bottom_indices.shape[0]
        bottom_indices[start_idx:end_idx, :actual_k] = batch_bottom_indices.T.cpu()
        bottom_values[start_idx:end_idx, :actual_k] = batch_bottom_values.T.cpu()
        
        # If we have fewer samples than top_k, pad with -1 for indices
        if actual_k < top_k:
            bottom_indices[start_idx:end_idx, actual_k:] = -1
            bottom_values[start_idx:end_idx, actual_k:] = float('inf')
    
    return bottom_indices, bottom_values


def save_results(
    top_indices: torch.Tensor, 
    top_values: torch.Tensor, 
    layer_name: str,
    output_dir: str,
    save_values: bool = False,
    metadata: Dict = None,
    input_file: str = None
):
    """Save top activation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create safe filename
    safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
    
    # Extract target_type from input filename if available
    target_type_suffix = ""
    if input_file:
        input_filename = os.path.basename(input_file)
        # Check if filename contains _input_ or _output_ pattern from extract_activations.py
        if "_input_" in input_filename:
            target_type_suffix = "_input"
        elif "_output_" in input_filename:
            target_type_suffix = "_output"
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(output_dir, f"top_activations_{safe_layer_name}{target_type_suffix}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    # Save as numpy for easy access
    numpy_path = os.path.join(output_dir, f"top_activations_{safe_layer_name}{target_type_suffix}_indices.npy")
    np.save(numpy_path, top_indices.numpy())
    print(f"Numpy indices saved to: {numpy_path}")
    
    if save_values:
        values_path = os.path.join(output_dir, f"top_activations_{safe_layer_name}{target_type_suffix}_values.npy")
        np.save(values_path, top_values.numpy())
        print(f"Numpy values saved to: {values_path}")


def analyze_results(top_indices: torch.Tensor, top_values: torch.Tensor, layer_name: str):
    """Analyze and print statistics about the results"""
    print(f"\n=== Analysis for {layer_name} ===")
    print(f"Shape: {top_indices.shape}")
    
    # Filter out invalid indices (-1)
    valid_mask = top_indices != -1
    valid_indices = top_indices[valid_mask]
    valid_values = top_values[valid_mask]
    
    if len(valid_indices) == 0:
        print("No valid activations found!")
        return
    
    print(f"Valid activations: {len(valid_indices)}")
    print(f"Value range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")
    print(f"Value statistics:")
    print(f"  Mean: {valid_values.mean():.6f}")
    print(f"  Std: {valid_values.std():.6f}")
    print(f"  Median: {valid_values.median():.6f}")
    
    # Analyze index distribution
    unique_indices, counts = torch.unique(valid_indices, return_counts=True)
    print(f"Unique sample indices: {len(unique_indices)}")
    print(f"Most frequently activated samples:")
    sorted_counts, sorted_idx = torch.sort(counts, descending=True)
    for i in range(min(10, len(sorted_counts))):
        sample_idx = unique_indices[sorted_idx[i]]
        count = sorted_counts[i]
        print(f"  Sample {sample_idx}: appears in {count} neurons")


def main():
    args = get_args()
    
    # Load activation file
    if args.input_file is not None:
        activations = load_activation_file(args.input_file)
    else:
        from dsets import get_imagenet
        from models import get_fn_model_loader
        from timm.data import create_transform, resolve_data_config
        from experiments.preprocessing.extract_activations import ActivationExtractor
        
        model = get_fn_model_loader(model_name=args.model_name)()
        model.eval().cuda();
        
        # Load dataset
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        transform = transform.transforms

        dataset = get_imagenet(
            data_path='/project/data/external/ILSVRC/Data/CLS-LOC',
            preprocessing=True,
            split='val',
            transform=transform
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Load activations
        extractor = ActivationExtractor(model, layers_to_hook=[(args.layer_name, 'output')])
        activations = extractor.extract(
            dataloader, 
            save_dir=f"/project/results/activations/imagenet/{args.model_name}", 
            save_intermediate=False,
            pool_type="raw"
        )
        
        safe_layer_name = args.layer_name.replace('.', '_')
        activations = load_file(f"/project/results/activations/imagenet/{args.model_name}/activations_{safe_layer_name}_output_raw.safetensors")
        

    # Process each layer in the file
    layer_name = args.layer_name
    activation_tensor = activations[layer_name]

    print(f"\n{'='*60}")
    print(f"Processing layer: {layer_name}")
    print(f"{'='*60}")
    
    # Compute top activations
    if args.mode == 'top':
        top_indices, top_values = compute_top_activations(
            activation_tensor,
            top_k=args.top_k,
            aggregation=args.aggregation,
            percentile=args.percentile,
            batch_size=args.batch_size,
            device=args.device,
            top_percentile=args.top_percentile
        )
    elif args.mode == 'middle':
        top_indices, top_values = compute_middle_activations(
            activation_tensor,
            top_k=args.top_k,
            aggregation=args.aggregation,
            percentile=args.percentile,
            batch_size=args.batch_size,
            device=args.device,
            top_percentile=args.top_percentile
        )
    elif args.mode == 'bottom':
        top_indices, top_values = compute_bottom_activations(
            activation_tensor,
            top_k=args.top_k,
            aggregation=args.aggregation,
            percentile=args.percentile,
            batch_size=args.batch_size,
            device=args.device,
            top_percentile=args.top_percentile
        )
    
    # Analyze results
    analyze_results(top_indices, top_values, layer_name)
    
    # Detect target type from input filename
    target_type = None
    if args.input_file:
        input_filename = os.path.basename(args.input_file)
        if "_input_" in input_filename:
            target_type = "input"
        elif "_output_" in input_filename:
            target_type = "output"
    
    # Prepare metadata
    metadata = {
        "layer_name": layer_name,
        "input_file": args.input_file,
        "target_type": target_type,
        "original_shape": list(activation_tensor.shape),
        "top_k": args.top_k,
        "aggregation": args.aggregation,
        "top_percentile": args.top_percentile if args.aggregation == "top_mean" else None,
        "percentile": args.percentile,
        "num_samples": activation_tensor.shape[0],
        "num_neurons": top_indices.shape[0],
        "valid_activations": int((top_indices != -1).sum()),
        "value_range": [float(top_values[top_indices != -1].min()), 
                        float(top_values[top_indices != -1].max())] if (top_indices != -1).any() else [0, 0]
    }
    
    # Save results
    save_results(
        top_indices,
        top_values,
        layer_name,
        args.output_dir,
        save_values=args.save_values,
        metadata=metadata,
        input_file=args.input_file
    )
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Usage Examples:

python /project/src/experiments/preprocessing/compute_top_activations.py \
    --model_name vit_b_16_timm \
    --layer_name "blocks.11" \
    --top_k 100 \
    --aggregation top_mean \
    --top_percentile 10.0 \
    --input_file /project/results/activations/imagenet/vit_b_16_timm/activations_blocks_11_output_raw.safetensors \
    --output_dir /project/results/top_activations/imagenet/vit_b_16_timm/top10pct \
    --save_values

python /project/src/experiments/preprocessing/compute_top_activations.py \
    --model_name vit_b_16_timm \
    --layer_name "blocks.11" \
    --top_k 50000 \
    --aggregation top_mean \
    --top_percentile 10.0 \
    --input_file /project/results/activations/imagenet/vit_b_16_timm/activations_blocks_11_output_raw.safetensors \
    --output_dir /project/results/top_activations/imagenet/vit_b_16_timm/top10pct_50000 \
    --save_values

python src/experiments/preprocessing/compute_top_activations.py \
    --input_file /project/results/activations/imagenet/imagenet/vit_b_16_timm/activations_blocks_7_output_raw.safetensors \
    --top_k 1000 \
    --aggregation top_mean \
    --top_percentile 10.0 \
    --output_dir results/top_activations/top10pct

python src/experiments/preprocessing/compute_top_activations.py \
    --input_file /project/results/activations/imagenet/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 1000 \
    --aggregation top_mean \
    --top_percentile 10.0 \
    --output_dir results/top_activations/top10pct
    

# Basic usage: Find top 100 activated samples for each neuron
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 100

# Use different aggregation for spatial dimensions
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 50 \
    --aggregation mean \
    --output_dir results/top_activations/mean_agg

# Use top 10% pixels mean for spatial aggregation
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 100 \
    --aggregation top_mean \
    --top_percentile 10.0

# Use top 5% pixels mean for more selective aggregation
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 50 \
    --aggregation top_mean \
    --top_percentile 5.0 \
    --output_dir results/top_activations/top5pct

# Only consider samples above 95th percentile
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 50 \
    --percentile 95 \
    --save_values

# Process with smaller batch size for memory efficiency
python src/experiments/preprocessing/compute_top_activations.py \
    --input_file results/activations/imagenet/vit_b_16_timm/activations_blocks_11_raw.safetensors \
    --top_k 200 \
    --batch_size 500 \
    --device cuda \
    --save_values

# Example loading results in Python:
import numpy as np
from safetensors import safe_open

# Load indices
indices = np.load("results/top_activations/top_activations_blocks_11_indices.npy")
print(f"Top activated sample indices shape: {indices.shape}")
print(f"Top 5 samples for neuron 0: {indices[0, :5]}")

# Load from safetensors
with safe_open("results/top_activations/top_activations_blocks_11.safetensors", framework="pt") as f:
    indices_tensor = f.get_tensor("blocks.11_top_indices")
    print(f"Indices shape: {indices_tensor.shape}")
""" 
