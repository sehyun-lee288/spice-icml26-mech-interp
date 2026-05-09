import argparse
import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import json
from safetensors import safe_open
from torchvision.transforms.functional import gaussian_blur

from dsets import get_dataset
from utils.helper import load_config


def get_args():
    parser = argparse.ArgumentParser(description="Crop image regions based on top activation indices and values")
    parser.add_argument(
        "--indices_file", 
        type=str, 
        required=True,
        help="Path to numpy file containing top activation indices"
    )
    parser.add_argument(
        "--values_file", 
        type=str, 
        default=None,
        help="Path to numpy file containing top activation values (optional)"
    )
    parser.add_argument(
        "--activation_file", 
        type=str, 
        required=True,
        help="Path to safetensors file containing original activations"
    )
    parser.add_argument(
        "--config_file", 
        type=str, 
        required=True,
        help="Path to configuration file for dataset/model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/cropped_regions",
        help="Directory to save cropped images"
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
        "--top_k_samples", 
        type=int, 
        default=10,
        help="Number of top samples to process per neuron"
    )
    parser.add_argument(
        "--crop_method", 
        type=str, 
        default="threshold",
        choices=["threshold", "bbox", "center"],
        help="Method for cropping: threshold (activation-based), bbox (bounding box), center (center crop)"
    )
    parser.add_argument(
        "--threshold_percentile", 
        type=float, 
        default=90.0,
        help="Percentile threshold for activation-based cropping"
    )
    parser.add_argument(
        "--crop_size", 
        type=int, 
        default=224,
        help="Size of cropped images (will be resized to this)"
    )
    parser.add_argument(
        "--padding", 
        type=int, 
        default=20,
        help="Padding around detected region"
    )
    parser.add_argument(
        "--save_overlay", 
        action="store_true",
        help="Save activation overlay on original image"
    )
    parser.add_argument(
        "--layer_name", 
        type=str, 
        default=None,
        help="Specific layer name to process (if not provided, will use first available)"
    )
    parser.add_argument(
        "--alpha_mask", 
        action="store_true",
        help="Create alpha mask with black background for non-activated regions"
    )
    parser.add_argument(
        "--mask_threshold", 
        type=float, 
        default=50.0,
        help="Percentile threshold for alpha mask (default: 50.0)"
    )
    
    return parser.parse_args()


def load_data(indices_file: str, values_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load top activation indices and values"""
    print(f"Loading indices from: {indices_file}")
    indices = np.load(indices_file)
    print(f"Indices shape: {indices.shape}")
    
    values = None
    if values_file and os.path.exists(values_file):
        print(f"Loading values from: {values_file}")
        values = np.load(values_file)
        print(f"Values shape: {values.shape}")
    
    return indices, values


def load_original_activations(activation_file: str, layer_name: Optional[str] = None) -> Tuple[torch.Tensor, str]:
    """Load original activation maps from safetensors file"""
    print(f"Loading original activations from: {activation_file}")
    
    with safe_open(activation_file, framework="pt", device="cpu") as f:
        available_keys = list(f.keys())
        print(f"Available layers: {available_keys}")
        
        # Select layer
        if layer_name and layer_name in available_keys:
            selected_layer = layer_name
        else:
            selected_layer = available_keys[0]
            print(f"Using layer: {selected_layer}")
        
        activations = f.get_tensor(selected_layer)
        print(f"Activation shape: {activations.shape}")
    
    return activations, selected_layer


def get_activation_map(activations: torch.Tensor, sample_idx: int, aggregation: str = "mean") -> np.ndarray:
    """Extract and process activation map for a specific sample"""
    if sample_idx >= activations.shape[0]:
        raise ValueError(f"Sample index {sample_idx} out of range (max: {activations.shape[0]-1})")
    
    # Get activation for specific sample
    sample_activation = activations[sample_idx]  # Shape: (channels, H, W) or (seq_len, hidden_dim)
    
    if sample_activation.dim() == 1:
        # For 1D activations (e.g., after global pooling), can't crop
        return None
    elif sample_activation.dim() == 2:
        # Handle ViT activations: reshape from sequence to spatial grid
        seq_len, hidden_dim = sample_activation.shape
        
        # Remove class token (first token) and get patch tokens
        num_patches = seq_len - 1
        
        # Find the square root to get spatial dimensions
        import math
        patch_size = int(math.sqrt(num_patches))
        
        if patch_size * patch_size == num_patches:
            print(f"Reshaping ViT: {seq_len} tokens -> {patch_size}x{patch_size} spatial grid")
            # Remove class token and reshape to spatial grid
            spatial_tokens = sample_activation[1:]  # Remove class token
            # Reshape to (patch_size, patch_size, hidden_dim) then to (hidden_dim, patch_size, patch_size)
            spatial_grid = spatial_tokens.reshape(patch_size, patch_size, hidden_dim)
            sample_activation = spatial_grid.permute(2, 0, 1)
            print(f"Reshaped to: {sample_activation.shape}")
        else:
            print(f"Warning: Could not reshape sequence of length {seq_len} to spatial grid")
            return None
    
    # Aggregate across channels
    if aggregation == "mean":
        activation_map = sample_activation.mean(dim=0)
    elif aggregation == "max":
        activation_map = sample_activation.max(dim=0)[0]
    elif aggregation == "sum":
        activation_map = sample_activation.sum(dim=0)
    elif aggregation == "raw":
        activation_map = sample_activation
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return activation_map.numpy()


def resize_activation_map(activation_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize activation map to match image size"""
    if activation_map is None:
        return None
    
    # Convert to tensor for interpolation
    activation_tensor = torch.from_numpy(activation_map).unsqueeze(0).unsqueeze(0).float()
    
    # Resize using bilinear interpolation
    resized = F.interpolate(
        activation_tensor, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    return resized.squeeze().numpy()


def get_crop_bbox(activation_map: np.ndarray, method: str, threshold_percentile: float = 90.0, 
                  padding: int = 20) -> Tuple[int, int, int, int]:
    """Get bounding box for cropping based on activation map"""
    H, W = activation_map.shape
    
    if method == "threshold":
        # Threshold-based cropping
        threshold = np.percentile(activation_map, threshold_percentile)
        mask = activation_map >= threshold
        
        # Find bounding box of activated region
        coords = np.where(mask)
        if len(coords[0]) == 0:
            # Fallback to center crop if no activations above threshold
            center_h, center_w = H // 2, W // 2
            crop_size = min(H, W) // 2
            y1, y2 = max(0, center_h - crop_size), min(H, center_h + crop_size)
            x1, x2 = max(0, center_w - crop_size), min(W, center_w + crop_size)
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add padding
            y1 = max(0, y_min - padding)
            y2 = min(H, y_max + padding)
            x1 = max(0, x_min - padding)
            x2 = min(W, x_max + padding)
    
    elif method == "bbox":
        # Use entire activation map to find bounding box
        # Find center of mass
        y_coords, x_coords = np.indices(activation_map.shape)
        total_activation = activation_map.sum()
        
        if total_activation > 0:
            center_y = (y_coords * activation_map).sum() / total_activation
            center_x = (x_coords * activation_map).sum() / total_activation
        else:
            center_y, center_x = H // 2, W // 2
        
        # Create bounding box around center
        crop_size = min(H, W) // 2
        y1 = max(0, int(center_y - crop_size))
        y2 = min(H, int(center_y + crop_size))
        x1 = max(0, int(center_x - crop_size))
        x2 = min(W, int(center_x + crop_size))
    
    elif method == "center":
        # Simple center crop
        center_h, center_w = H // 2, W // 2
        crop_size = min(H, W) // 2
        y1, y2 = max(0, center_h - crop_size), min(H, center_h + crop_size)
        x1, x2 = max(0, center_w - crop_size), min(W, center_w + crop_size)
    
    return x1, y1, x2, y2


def crop_and_resize_image(image: Image.Image, bbox: Tuple[int, int, int, int], 
                         target_size: int) -> Image.Image:
    """Crop and resize image based on bounding box"""
    x1, y1, x2, y2 = bbox
    
    # Crop image
    cropped = image.crop((x1, y1, x2, y2))
    
    # Resize to target size
    resized = cropped.resize((target_size, target_size), Image.LANCZOS)
    
    return resized


def create_alpha_mask_crop(image: Image.Image, activation_map: np.ndarray, bbox: Tuple[int, int, int, int], 
                          target_size: int, threshold_percentile: float = 50.0) -> Image.Image:
    """Create cropped image with alpha mask based on activation map"""
    print(f"create_alpha_mask_crop called with activation_map shape: {activation_map.shape}")
    x1, y1, x2, y2 = bbox
    
    # Crop image
    cropped_image = image.crop((x1, y1, x2, y2))
    
    # Crop activation map
    cropped_activation = activation_map[y1:y2, x1:x2]
    
    # Resize both image and activation map
    resized_image = cropped_image.resize((target_size, target_size), Image.LANCZOS)
    
    # Resize activation map
    activation_tensor = torch.from_numpy(cropped_activation).unsqueeze(0).unsqueeze(0).float()
    resized_activation = F.interpolate(
        activation_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()
    
    # Add Gaussian blur (kernel_size must be odd and > 0)
    blurred_activation = gaussian_blur(torch.from_numpy(resized_activation).unsqueeze(0).unsqueeze(0), kernel_size=[51, 51])
    resized_activation = blurred_activation.squeeze().numpy()
    
    # Create alpha mask based on activation threshold
    activation_threshold = np.percentile(resized_activation, threshold_percentile)
    alpha_mask = (resized_activation >= activation_threshold).astype(np.uint8) * 255
    
    # Debug info
    pixels_kept = np.sum(alpha_mask > 0)
    total_pixels = alpha_mask.size
    print(f"Alpha mask: {pixels_kept}/{total_pixels} pixels kept ({pixels_kept/total_pixels*100:.1f}%) with threshold {threshold_percentile}%")
    
    # Convert to PIL Image
    alpha_mask_pil = Image.fromarray(alpha_mask, mode='L')
    
    # Convert image to RGBA if not already
    if resized_image.mode != 'RGBA':
        resized_image = resized_image.convert('RGBA')
    
    # Create black background
    black_background = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 255))
    
    # Apply alpha mask - only keep activated regions
    masked_image = Image.composite(resized_image, black_background, alpha_mask_pil)
    print(f"Alpha mask crop completed - image mode: {masked_image.mode}")
    
    return masked_image


def create_activation_overlay(image: Image.Image, activation_map: np.ndarray, 
                            alpha: float = 0.4) -> Image.Image:
    """Create overlay of activation map on original image"""
    # Normalize activation map
    activation_norm = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    
    # Convert to heatmap
    activation_colored = cv2.applyColorMap((activation_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    activation_colored = cv2.cvtColor(activation_colored, cv2.COLOR_BGR2RGB)
    activation_img = Image.fromarray(activation_colored)
    
    # Resize to match image
    activation_img = activation_img.resize(image.size, Image.LANCZOS)
    
    # Create overlay
    overlay = Image.blend(image, activation_img, alpha)
    
    return overlay


def save_image_with_info(image: Image.Image, save_path: str, info: Dict):
    """Save image with metadata"""
    # Save image
    image.save(save_path)
    
    # Save metadata
    info_path = save_path.replace('.jpg', '_info.json').replace('.png', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def get_cropped_images(
    sample_indices: List[int],
    neuron_idx: int,
    model: torch.nn.Module,
    layer_name: str,
    config_file: str,
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = False,
    mask_threshold: float = 50.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32
) -> List[Tuple[Image.Image, Dict]]:
    """
    Get cropped images for specified sample indices and neuron index by computing activations directly.
    
    Args:
        sample_indices: List of sample indices to process
        neuron_idx: Index of the neuron to use for activation-based cropping
        model: PyTorch model to use for computing activations
        layer_name: Name of the layer to extract activations from
        config_file: Path to configuration file for dataset/model
        crop_method: Method for cropping ("threshold", "bbox", "center")
        threshold_percentile: Percentile threshold for activation-based cropping
        crop_size: Size of cropped images (will be resized to this)
        padding: Padding around detected region
        alpha_mask: Create alpha mask with black background for non-activated regions
        mask_threshold: Percentile threshold for alpha mask
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for processing samples
        
    Returns:
        List of tuples (cropped_image, metadata_dict) for each sample
    """
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Load configuration and dataset
    config = load_config(config_file)
    dataset_name = config["dataset_name"]
    data_path = config.get("data_path", None)
    
    # Setup dataset with preprocessing for model inference
    dataset_processed = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=True,  # Get preprocessed images for model
        split="val",
        transform=None,
    )
    
    # Setup dataset without preprocessing to get original images for cropping
    dataset_original = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=False,  # Get original images
        split="val",
        transform=None,
    )
    
    print(f"Dataset loaded: {len(dataset_processed)} samples")
    
    # Setup activation hook
    activations = {}
    
    def get_activation_hook(name):
        def hook(model, input, output):
            if isinstance(output, list):
                assert len(output) == 1, f"HARD CODE"
                output = output[0]
            activations[name] = output.detach()
        return hook
    
    # Register hook
    hook_handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(get_activation_hook(layer_name))
            break
    
    if hook_handle is None:
        raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {[name for name, _ in model.named_modules()]}")
    
    print(f"Registered hook for layer: {layer_name}")
    
    # Process samples in batches to compute activations
    all_activations = []
    original_images = []
    labels = []
    
    try:
        with torch.no_grad():
            for i in range(0, len(sample_indices), batch_size):
                batch_indices = sample_indices[i:i + batch_size]
                
                # Get preprocessed images for model
                batch_images = []
                batch_originals = []
                batch_labels = []
                
                for idx in batch_indices:
                    if idx >= len(dataset_processed):
                        print(f"Warning: Sample index {idx} out of range (max: {len(dataset_processed)-1}), skipping")
                        continue
                    
                    # Get preprocessed image for model
                    proc_image, label = dataset_processed[idx]
                    # Get original image for cropping
                    orig_image, _ = dataset_original[idx]
                    
                    batch_images.append(proc_image)
                    batch_originals.append(orig_image)
                    batch_labels.append(label)
                
                if not batch_images:
                    continue
                
                # Stack images into batch tensor
                batch_tensor = torch.stack(batch_images).to(device)
                
                # Forward pass to get activations
                _ = model(batch_tensor)
                
                # Extract activations for this batch
                if layer_name in activations:
                    batch_activations = activations[layer_name].cpu()
                    all_activations.append(batch_activations)
                    original_images.extend(batch_originals)
                    labels.extend(batch_labels)
                else:
                    print(f"Warning: No activations captured for layer {layer_name}")
        
        # Remove hook
        if hook_handle:
            hook_handle.remove()
        
        if not all_activations:
            raise ValueError(f"No activations were captured for layer {layer_name}")
        
        # Concatenate all activations
        all_activations_tensor = torch.cat(all_activations, dim=0)
        print(f"Computed activations shape: {all_activations_tensor.shape}")
        
        # Check if layer has spatial activations
        sample_activation = all_activations_tensor[0]  # Check first sample
        has_spatial_activations = True
        
        if sample_activation.dim() == 1:
            print(f"WARNING: Layer '{layer_name}' has 1D activations (shape: {sample_activation.shape})")
            print("This layer doesn't have spatial dimensions - will use center crop instead of activation-based cropping")
            has_spatial_activations = False
        elif sample_activation.dim() == 2:
            seq_len, hidden_dim = sample_activation.shape
            if 'dinovit' in type(model).__name__.lower():
                num_patches = seq_len - 1 - 4 # 4 for regsiter token
            else:
                num_patches = seq_len - 1
            import math
            patch_size = int(math.sqrt(num_patches))
            
            if patch_size * patch_size == num_patches:
                print(f"Layer '{layer_name}' has ViT activations (shape: {sample_activation.shape})")
                print(f"Will reshape from sequence to spatial grid: {patch_size}x{patch_size}")
            else:
                print(f"WARNING: Layer '{layer_name}' has 2D activations (shape: {sample_activation.shape})")
                print("Cannot reshape to spatial grid - will use center crop instead of activation-based cropping")
                has_spatial_activations = False
        elif sample_activation.dim() == 3: # CNN
            channel, width, height = sample_activation.shape
        else:
            print(f"Layer '{layer_name}' has spatial activations (shape: {sample_activation.shape})")
        
        # Validate inputs
        if all_activations_tensor.ndim == 3:
            if neuron_idx >= all_activations_tensor.shape[2]:
                raise ValueError(f"Neuron index {neuron_idx} out of range (max: {all_activations_tensor.shape[1]-1})")
        elif all_activations_tensor.ndim == 4:
            if neuron_idx >= all_activations_tensor.shape[1]:
                raise ValueError(f"Neuron index {neuron_idx} out of range (max: {all_activations_tensor.shape[1]-1})")
        
        results = []
        
        # Process each sample
        for i, (image, label) in enumerate(zip(original_images, labels)):
            try:
                # Convert image to PIL if needed
                if not isinstance(image, Image.Image):
                    # Convert tensor to numpy array if needed
                    if hasattr(image, 'numpy'):
                        image = image.numpy()
                    elif isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy()
                    
                    # Handle different tensor formats
                    if image.ndim == 3 and image.shape[0] == 3:
                        # CHW format, convert to HWC
                        image = image.transpose(1, 2, 0)
                    elif image.ndim == 4 and image.shape[0] == 1:
                        # BCHW format with batch size 1, convert to HWC
                        image = image.squeeze(0).transpose(1, 2, 0)
                    
                    # Ensure values are in [0, 255] range
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                    
                    image = Image.fromarray(image)
                
                # Get activation map for this sample
                activation_map = get_activation_map(all_activations_tensor, i, aggregation="raw")
                
                if activation_map is None or not has_spatial_activations:
                    # For layers without spatial dimensions (1D activations), do center crop
                    print(f"No spatial activations for sample {sample_indices[i]}, using center crop")
                    
                    # Default to center crop
                    W, H = image.size
                    crop_size_pixels = min(W, H) // 2
                    center_x, center_y = W // 2, H // 2
                    x1 = max(0, center_x - crop_size_pixels)
                    y1 = max(0, center_y - crop_size_pixels)
                    x2 = min(W, center_x + crop_size_pixels)
                    y2 = min(H, center_y + crop_size_pixels)
                    bbox = (x1, y1, x2, y2)
                    
                    # Crop and resize image
                    cropped_image = crop_and_resize_image(image, bbox, crop_size)
                    
                    # No activation overlay for 1D activations
                    resized_activation = None
                    
                else:
                    # Extract specific neuron's activation map
                    neuron_activation_map = activation_map[neuron_idx]
                    
                    # Resize activation map to match image size
                    resized_activation = resize_activation_map(neuron_activation_map, image.size[::-1])  # PIL size is (W, H)
                    
                    # Get crop bounding box
                    bbox = get_crop_bbox(
                        resized_activation, 
                        crop_method, 
                        threshold_percentile, 
                        padding
                    )
                    
                    # Crop and resize image (with alpha mask if requested)
                    if alpha_mask:
                        cropped_image = create_alpha_mask_crop(
                            image, resized_activation, bbox, crop_size, mask_threshold
                        )
                    else:
                        cropped_image = crop_and_resize_image(image, bbox, crop_size)
                
                # Prepare metadata (convert numpy types to Python types for JSON serialization)
                metadata = {
                    "neuron_idx": int(neuron_idx),
                    "sample_idx": int(sample_indices[i]),
                    "label": int(label) if isinstance(label, (int, np.integer)) else str(label),
                    "layer_name": layer_name,
                    "crop_bbox": [int(x) for x in bbox],  # Convert bbox coordinates to int
                    "crop_method": "center_crop" if resized_activation is None else crop_method,
                    "has_spatial_activations": resized_activation is not None,
                    "alpha_mask": alpha_mask,
                    "mask_threshold": mask_threshold if alpha_mask else None,
                    "original_size": [int(x) for x in image.size],
                    "cropped_size": (crop_size, crop_size),
                    "threshold_percentile": threshold_percentile,
                    "padding": padding
                }
                
                results.append((cropped_image, metadata))
                
            except Exception as e:
                print(f"Error processing sample {sample_indices[i] if i < len(sample_indices) else i}: {e}")
                continue
    
    except Exception as e:
        # Make sure to remove hook even if there's an error
        if hook_handle:
            hook_handle.remove()
        raise e
    
    return results


def get_cropped_images_fast(
    sample_indices: List[int],
    neuron_idx: int,
    model: torch.nn.Module,
    layer_name: str,
    config_file: str,
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = False,
    mask_threshold: float = 50.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64
) -> List[Tuple[Image.Image, Dict]]:
    """
    Optimized version of get_cropped_images that pre-computes all activations at once.
    Much faster than the original version for multiple samples.
    
    Args:
        sample_indices: List of sample indices to process
        neuron_idx: Index of the neuron to use for activation-based cropping
        model: PyTorch model to use for computing activations
        layer_name: Name of the layer to extract activations from
        config_file: Path to configuration file for dataset/model
        crop_method: Method for cropping ("threshold", "bbox", "center")
        threshold_percentile: Percentile threshold for activation-based cropping
        crop_size: Size of cropped images (will be resized to this)
        padding: Padding around detected region
        alpha_mask: Create alpha mask with black background for non-activated regions
        mask_threshold: Percentile threshold for alpha mask
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for processing samples (larger = faster but more memory)
        
    Returns:
        List of tuples (cropped_image, metadata_dict) for each sample
    """
    print(f"Processing {len(sample_indices)} samples with optimized function...")
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Load configuration and dataset
    config = load_config(config_file)
    dataset_name = config["dataset_name"]
    data_path = config.get("data_path", None)
    
    # Setup datasets
    dataset_processed = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=True,
        split="val",
        transform=None,
    )
    
    dataset_original = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=False,
        split="val",
        transform=None,
    )
    
    # Filter valid sample indices
    valid_sample_indices = [idx for idx in sample_indices if idx < len(dataset_processed)]
    if len(valid_sample_indices) != len(sample_indices):
        print(f"Warning: {len(sample_indices) - len(valid_sample_indices)} invalid sample indices filtered out")
    
    # Pre-compute all activations at once
    print("Pre-computing activations for all samples...")
    all_activations = []
    original_images = []
    labels = []
    
    # Setup activation hook
    activations_cache = {}
    
    def get_activation_hook(name):
        def hook(model, input, output):
            activations_cache[name] = output.detach().cpu()
        return hook
    
    # Register hook
    hook_handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(get_activation_hook(layer_name))
            break
    
    if hook_handle is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    try:
        with torch.no_grad():
            # Process in larger batches for efficiency
            for i in range(0, len(valid_sample_indices), batch_size):
                batch_indices = valid_sample_indices[i:i + batch_size]
                
                # Prepare batch
                batch_images = []
                batch_originals = []
                batch_labels = []
                
                for idx in batch_indices:
                    proc_image, label = dataset_processed[idx]
                    orig_image, _ = dataset_original[idx]
                    
                    batch_images.append(proc_image)
                    batch_originals.append(orig_image)
                    batch_labels.append(label)
                
                # Forward pass
                batch_tensor = torch.stack(batch_images).to(device)
                _ = model(batch_tensor)
                
                # Store results
                if layer_name in activations_cache:
                    all_activations.append(activations_cache[layer_name])
                    original_images.extend(batch_originals)
                    labels.extend(batch_labels)
                    
                # Clear cache for memory efficiency
                activations_cache.clear()
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {min(i + batch_size, len(valid_sample_indices))}/{len(valid_sample_indices)} samples")
        
        # Remove hook
        hook_handle.remove()
        
        # Concatenate all activations
        all_activations_tensor = torch.cat(all_activations, dim=0)
        print(f"Computed activations shape: {all_activations_tensor.shape}")
        
    except Exception as e:
        if hook_handle:
            hook_handle.remove()
        raise e
    
    # Check activation format
    sample_activation = all_activations_tensor[0]
    has_spatial_activations = True
    
    if sample_activation.dim() == 1:
        print(f"WARNING: Layer has 1D activations - using center crop")
        has_spatial_activations = False
    elif sample_activation.dim() == 2:
        seq_len, hidden_dim = sample_activation.shape
        if 'dinovit' in type(model).__name__.lower():
            num_patches = seq_len - 1 - 4 # 4 for register token
        else:
            num_patches = seq_len - 1
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            print(f"WARNING: Cannot reshape sequence to spatial grid - using center crop")
            has_spatial_activations = False
    
    # Validate neuron index
    if neuron_idx >= all_activations_tensor.shape[1]:
        raise ValueError(f"Neuron index {neuron_idx} out of range (max: {all_activations_tensor.shape[1]-1})")
    
    # Process all samples
    print("Processing crops...")
    results = []
    
    for i, (image, label, orig_sample_idx) in enumerate(zip(original_images, labels, valid_sample_indices)):
        try:
            # Convert image to PIL if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                elif isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                
                if image.ndim == 3 and image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                elif image.ndim == 4 and image.shape[0] == 1:
                    image = image.squeeze(0).transpose(1, 2, 0)
                
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                image = Image.fromarray(image)
            
            if not has_spatial_activations:
                # Center crop for non-spatial activations
                W, H = image.size
                crop_size_pixels = min(W, H) // 2
                center_x, center_y = W // 2, H // 2
                x1 = max(0, center_x - crop_size_pixels)
                y1 = max(0, center_y - crop_size_pixels)
                x2 = min(W, center_x + crop_size_pixels)
                y2 = min(H, center_y + crop_size_pixels)
                bbox = (x1, y1, x2, y2)
                
                cropped_image = crop_and_resize_image(image, bbox, crop_size)
                
            else:
                # Activation-based crop
                activation_map = get_activation_map(all_activations_tensor, i, aggregation="raw")
                neuron_activation_map = activation_map[neuron_idx]
                
                # Resize activation map to match image size
                resized_activation = resize_activation_map(neuron_activation_map, image.size[::-1])
                
                # Get crop bounding box
                bbox = get_crop_bbox(
                    resized_activation, 
                    crop_method, 
                    threshold_percentile, 
                    padding
                )
                
                # Crop and resize
                if alpha_mask:
                    cropped_image = create_alpha_mask_crop(
                        image, resized_activation, bbox, crop_size, mask_threshold
                    )
                else:
                    cropped_image = crop_and_resize_image(image, bbox, crop_size)
            
            # Metadata
            metadata = {
                "neuron_idx": int(neuron_idx),
                "sample_idx": int(orig_sample_idx),
                "label": int(label) if isinstance(label, (int, np.integer)) else str(label),
                "layer_name": layer_name,
                "crop_bbox": [int(x) for x in bbox],
                "crop_method": "center_crop" if not has_spatial_activations else crop_method,
                "has_spatial_activations": has_spatial_activations,
                "alpha_mask": alpha_mask,
                "mask_threshold": mask_threshold if alpha_mask else None,
                "original_size": [int(x) for x in image.size],
                "cropped_size": (crop_size, crop_size),
                "threshold_percentile": threshold_percentile,
                "padding": padding
            }
            
            results.append((cropped_image, metadata))
            
        except Exception as e:
            print(f"Error processing sample {orig_sample_idx}: {e}")
            continue
    
    print(f"Successfully processed {len(results)}/{len(valid_sample_indices)} samples")
    return results


def get_cropped_images_cached(
    sample_indices: List[int],
    neuron_indices: List[int],
    model: torch.nn.Module,
    layer_name: str,
    config_file: str,
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = False,
    mask_threshold: float = 50.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64
) -> Dict[int, List[Tuple[Image.Image, Dict]]]:
    """
    Most efficient version: compute activations once and process multiple neurons.
    
    Args:
        sample_indices: List of sample indices to process
        neuron_indices: List of neuron indices to process
        model: PyTorch model
        layer_name: Layer name
        config_file: Config file path
        ... other parameters
        
    Returns:
        Dict mapping neuron_idx -> List of (cropped_image, metadata) tuples
    """
    print(f"Processing {len(sample_indices)} samples for {len(neuron_indices)} neurons...")
    
    # Compute activations once for all samples
    first_results = get_cropped_images_fast(
        sample_indices=sample_indices,
        neuron_idx=neuron_indices[0],  # Use first neuron to get activations
        model=model,
        layer_name=layer_name,
        config_file=config_file,
        crop_method=crop_method,
        threshold_percentile=threshold_percentile,
        crop_size=crop_size,
        padding=padding,
        alpha_mask=alpha_mask,
        mask_threshold=mask_threshold,
        device=device,
        batch_size=batch_size
    )
    
    # Store results for first neuron
    results = {neuron_indices[0]: first_results}
    
    # For remaining neurons, reuse the activation computation logic but with cached activations
    # (This would require refactoring to separate activation computation from cropping)
    
    return results


def get_cropped_images_from_top_activations(
    indices_file: str,
    neuron_idx: int,
    model: torch.nn.Module,
    layer_name: str,
    config_file: str,
    top_k_samples: int = 10,
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = False,
    mask_threshold: float = 50.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32
) -> List[Tuple[Image.Image, Dict]]:
    """
    Get cropped images for top-k samples of a specific neuron using pre-computed top activation indices.
    
    Args:
        indices_file: Path to numpy file containing top activation indices
        neuron_idx: Index of the neuron to process
        model: PyTorch model to use for computing activations
        layer_name: Name of the layer to extract activations from
        config_file: Path to configuration file for dataset/model
        top_k_samples: Number of top samples to process
        crop_method: Method for cropping ("threshold", "bbox", "center")
        threshold_percentile: Percentile threshold for activation-based cropping
        crop_size: Size of cropped images (will be resized to this)
        padding: Padding around detected region
        alpha_mask: Create alpha mask with black background for non-activated regions
        mask_threshold: Percentile threshold for alpha mask
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for processing samples
        
    Returns:
        List of tuples (cropped_image, metadata_dict) for each sample
    """
    # Load top activation indices
    indices, _ = load_data(indices_file, None)
    
    # Validate neuron index
    if neuron_idx >= indices.shape[0]:
        raise ValueError(f"Neuron index {neuron_idx} out of range (max: {indices.shape[0]-1})")
    
    # Get top samples for this neuron
    neuron_indices = indices[neuron_idx, :top_k_samples]
    
    # Filter out invalid indices
    valid_indices = [int(idx) for idx in neuron_indices if idx != -1]
    
    # Get cropped images for these samples
    return get_cropped_images(
        sample_indices=valid_indices,
        neuron_idx=neuron_idx,
        model=model,
        layer_name=layer_name,
        config_file=config_file,
        crop_method=crop_method,
        threshold_percentile=threshold_percentile,
        crop_size=crop_size,
        padding=padding,
        alpha_mask=alpha_mask,
        mask_threshold=mask_threshold,
        device=device,
        batch_size=batch_size
    )


def get_cropped_images_from_activations(
    sample_indices: List[int],
    neuron_idx: int,
    activations: torch.Tensor,
    config_file: str,
    layer_name: str = "unknown",
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = False,
    mask_threshold: float = 50.0
) -> List[Tuple[Image.Image, Dict]]:
    """
    FASTEST VERSION: Use pre-computed activations directly.
    Perfect for when you already have activations loaded in memory.
    
    Args:
        sample_indices: List of sample indices to process
        neuron_idx: Index of the neuron to use for activation-based cropping
        activations: Pre-computed activation tensor [samples, neurons, ...] or [samples, height, width]
        config_file: Path to configuration file for dataset/model
        layer_name: Name of the layer (for metadata only)
        crop_method: Method for cropping ("threshold", "bbox", "center")
        threshold_percentile: Percentile threshold for activation-based cropping
        crop_size: Size of cropped images (will be resized to this)
        padding: Padding around detected region
        alpha_mask: Create alpha mask with black background for non-activated regions
        mask_threshold: Percentile threshold for alpha mask
        
    Returns:
        List of tuples (cropped_image, metadata_dict) for each sample
    """
    print(f"Processing {len(sample_indices)} samples with pre-computed activations (fastest)...")
    
    # Load configuration and dataset
    config = load_config(config_file)
    dataset_name = config["dataset_name"]
    data_path = config.get("data_path", None)
    
    # Setup dataset for original images only
    dataset_original = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=False,
        split="val",
        transform=None,
    )
    
    # Filter valid sample indices
    valid_sample_indices = [idx for idx in sample_indices if idx < len(dataset_original) and idx < activations.shape[0]]
    if len(valid_sample_indices) != len(sample_indices):
        print(f"Warning: {len(sample_indices) - len(valid_sample_indices)} invalid sample indices filtered out")
    
    # Check activation format
    sample_activation = activations[0]
    has_spatial_activations = True
    
    if sample_activation.dim() == 1:
        print(f"WARNING: Layer has 1D activations - using center crop")
        has_spatial_activations = False
    elif sample_activation.dim() == 2:
        seq_len, hidden_dim = sample_activation.shape
        if 'dinovit' in type(model).__name__.lower():
            num_patches = seq_len - 1 - 4 # 4 for register token
        else:
            num_patches = seq_len - 1
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            print(f"WARNING: Cannot reshape sequence to spatial grid - using center crop")
            has_spatial_activations = False
    
    # Validate neuron index
    if has_spatial_activations and neuron_idx >= activations.shape[1]:
        raise ValueError(f"Neuron index {neuron_idx} out of range (max: {activations.shape[1]-1})")
    
    results = []
    
    # Process each sample
    for sample_idx in valid_sample_indices:
        try:
            # Get original image
            image, label = dataset_original[sample_idx]
            
            # Convert image to PIL if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                elif isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                
                if image.ndim == 3 and image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                elif image.ndim == 4 and image.shape[0] == 1:
                    image = image.squeeze(0).transpose(1, 2, 0)
                
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                image = Image.fromarray(image)
            
            if not has_spatial_activations:
                # Center crop for non-spatial activations
                W, H = image.size
                crop_size_pixels = min(W, H) // 2
                center_x, center_y = W // 2, H // 2
                x1 = max(0, center_x - crop_size_pixels)
                y1 = max(0, center_y - crop_size_pixels)
                x2 = min(W, center_x + crop_size_pixels)
                y2 = min(H, center_y + crop_size_pixels)
                bbox = (x1, y1, x2, y2)
                
                cropped_image = crop_and_resize_image(image, bbox, crop_size)
                
            else:
                # Get activation map for this sample
                activation_map = get_activation_map(activations, sample_idx, aggregation="raw")
                
                if activation_map is None:
                    # Fallback to center crop
                    W, H = image.size
                    crop_size_pixels = min(W, H) // 2
                    center_x, center_y = W // 2, H // 2
                    x1 = max(0, center_x - crop_size_pixels)
                    y1 = max(0, center_y - crop_size_pixels)
                    x2 = min(W, center_x + crop_size_pixels)
                    y2 = min(H, center_y + crop_size_pixels)
                    bbox = (x1, y1, x2, y2)
                    
                    cropped_image = crop_and_resize_image(image, bbox, crop_size)
                else:
                    # Extract specific neuron's activation map
                    neuron_activation_map = activation_map[neuron_idx]
                    
                    # Resize activation map to match image size
                    resized_activation = resize_activation_map(neuron_activation_map, image.size[::-1])
                    
                    # Get crop bounding box
                    bbox = get_crop_bbox(
                        resized_activation, 
                        crop_method, 
                        threshold_percentile, 
                        padding
                    )
                    
                    # Crop and resize
                    if alpha_mask:
                        cropped_image = create_alpha_mask_crop(
                            image, resized_activation, bbox, crop_size, mask_threshold
                        )
                    else:
                        cropped_image = crop_and_resize_image(image, bbox, crop_size)
            
            # Metadata
            metadata = {
                "neuron_idx": int(neuron_idx),
                "sample_idx": int(sample_idx),
                "label": int(label) if isinstance(label, (int, np.integer)) else str(label),
                "layer_name": layer_name,
                "crop_bbox": [int(x) for x in bbox],
                "crop_method": "center_crop" if not has_spatial_activations else crop_method,
                "has_spatial_activations": has_spatial_activations,
                "alpha_mask": alpha_mask,
                "mask_threshold": mask_threshold if alpha_mask else None,
                "original_size": [int(x) for x in image.size],
                "cropped_size": (crop_size, crop_size),
                "threshold_percentile": threshold_percentile,
                "padding": padding
            }
            
            results.append((cropped_image, metadata))
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"Successfully processed {len(results)}/{len(valid_sample_indices)} samples")
    return results


def main():
    args = get_args()
    
    # Load data
    indices, values = load_data(args.indices_file, args.values_file)
    activations, layer_name = load_original_activations(args.activation_file, args.layer_name)
    
    # Check if layer has spatial activations
    sample_activation = activations[0]  # Check first sample
    if sample_activation.dim() == 1:
        print(f"WARNING: Layer '{layer_name}' has 1D activations (shape: {sample_activation.shape})")
        print("This layer doesn't have spatial dimensions - will use center crop instead of activation-based cropping")
    elif sample_activation.dim() == 2:
        seq_len, hidden_dim = sample_activation.shape
        if 'dinovit' in type(args.config_file).__name__.lower():
            num_patches = seq_len - 1 - 4 # 4 for register token
        else:
            num_patches = seq_len - 1
        import math
        patch_size = int(math.sqrt(num_patches))
        
        if patch_size * patch_size == num_patches:
            print(f"Layer '{layer_name}' has ViT activations (shape: {sample_activation.shape})")
            print(f"Will reshape from sequence to spatial grid: {patch_size}x{patch_size}")
        else:
            print(f"WARNING: Layer '{layer_name}' has 2D activations (shape: {sample_activation.shape})")
            print("Cannot reshape to spatial grid - will use center crop instead of activation-based cropping")
    else:
        print(f"Layer '{layer_name}' has spatial activations (shape: {sample_activation.shape})")
    
    # Load configuration and dataset
    config = load_config(args.config_file)
    dataset_name = config["dataset_name"]
    data_path = config.get("data_path", None)
    
    # Setup dataset (without transforms to get original images)
    dataset = get_dataset(dataset_name)(
        data_path=data_path,
        preprocessing=False,  # Get original images
        split="val",
        transform=None,
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which neurons to process
    if args.all_neurons:
        # Process all available neurons
        neuron_indices_to_process = list(range(indices.shape[0]))
        print(f"Processing all {len(neuron_indices_to_process)} neurons")
    elif args.neuron_indices is not None:
        # Process specified neurons
        neuron_indices_to_process = args.neuron_indices
        print(f"Processing {len(neuron_indices_to_process)} specified neurons: {neuron_indices_to_process}")
    else:
        # Default: process first 5 neurons
        neuron_indices_to_process = [0, 1, 2, 3, 4]
        print(f"Processing default neurons: {neuron_indices_to_process}")
    
    # Process each neuron
    for neuron_idx in tqdm(neuron_indices_to_process, desc="Processing neurons"):
        if neuron_idx >= indices.shape[0]:
            print(f"Skipping neuron {neuron_idx} (out of range)")
            continue
        
        print(f"\nProcessing neuron {neuron_idx}")
        neuron_dir = os.path.join(args.output_dir, f"neuron_{neuron_idx:03d}")
        os.makedirs(neuron_dir, exist_ok=True)
        
        # Get top samples for this neuron
        neuron_indices = indices[neuron_idx, :args.top_k_samples]
        neuron_values = values[neuron_idx, :args.top_k_samples] if values is not None else None
        
        # Process each sample
        for rank, sample_idx in enumerate(neuron_indices):
            if sample_idx == -1:  # Invalid index
                continue
                
            try:
                # Get image and label
                image, label = dataset[sample_idx]
                if not isinstance(image, Image.Image):
                    # Convert tensor to numpy array if needed
                    if hasattr(image, 'numpy'):
                        image = image.numpy()
                    elif isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy()
                    
                    # Handle different tensor formats
                    if image.ndim == 3 and image.shape[0] == 3:
                        # CHW format, convert to HWC
                        image = image.transpose(1, 2, 0)
                    elif image.ndim == 4 and image.shape[0] == 1:
                        # BCHW format with batch size 1, convert to HWC
                        image = image.squeeze(0).transpose(1, 2, 0)
                    
                    # Ensure values are in [0, 255] range
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                    
                    image = Image.fromarray(image)
                
                # Get activation map for this sample
                print(f"Getting activation map for sample {sample_idx}...")
                activation_map = get_activation_map(activations, sample_idx, aggregation="raw")
                print(f"Raw activation map: {activation_map.shape if activation_map is not None else 'None'}")
                
                if activation_map is None:
                    # For layers without spatial dimensions (1D activations), do center crop
                    print(f"No spatial activations for sample {sample_idx}, using center crop (alpha mask disabled)")
                    
                    # Default to center crop
                    W, H = image.size
                    crop_size = min(W, H) // 2
                    center_x, center_y = W // 2, H // 2
                    x1 = max(0, center_x - crop_size)
                    y1 = max(0, center_y - crop_size)
                    x2 = min(W, center_x + crop_size)
                    y2 = min(H, center_y + crop_size)
                    bbox = (x1, y1, x2, y2)
                    
                    # Crop and resize image
                    print(f"Using center crop (no spatial activations available)")
                    cropped_image = crop_and_resize_image(image, bbox, args.crop_size)
                    
                    # No activation overlay for 1D activations
                    resized_activation = None
                    
                    cropped_image_without_alpha_mask = None
                else:
                    # Extract specific neuron's activation map
                    neuron_activation_map = activation_map[neuron_idx]
                    print(f"Neuron {neuron_idx} activation map shape: {neuron_activation_map.shape}")
                    
                    # Resize activation map to match image size
                    resized_activation = resize_activation_map(neuron_activation_map, image.size[::-1])  # PIL size is (W, H)
                    
                    # Get crop bounding box
                    bbox = get_crop_bbox(
                        resized_activation, 
                        args.crop_method, 
                        args.threshold_percentile, 
                        args.padding
                    )
                    
                    # Crop and resize image (with alpha mask if requested)
                    if args.alpha_mask:
                        print(f"Applying alpha mask with threshold {args.mask_threshold}%")
                        cropped_image = create_alpha_mask_crop(
                            image, resized_activation, bbox, args.crop_size, args.mask_threshold
                        )
                        
                    cropped_image_without_alpha_mask = crop_and_resize_image(image, bbox, args.crop_size)
                
                # Prepare metadata (convert numpy types to Python types for JSON serialization)
                info = {
                    "neuron_idx": int(neuron_idx),
                    "sample_idx": int(sample_idx),
                    "rank": int(rank),
                    "activation_value": float(neuron_values[rank]) if neuron_values is not None else None,
                    "label": int(label) if isinstance(label, (int, np.integer)) else str(label),
                    "layer_name": layer_name,
                    "crop_bbox": [int(x) for x in bbox],  # Convert bbox coordinates to int
                    "crop_method": "center_crop" if resized_activation is None else args.crop_method,
                    "has_spatial_activations": resized_activation is not None,
                    "alpha_mask": args.alpha_mask,
                    "mask_threshold": args.mask_threshold if args.alpha_mask else None,
                    "original_size": [int(x) for x in image.size],
                    "cropped_size": (args.crop_size, args.crop_size)
                }
                
                # Save cropped image (PNG for alpha mask, JPG for regular)
                if args.alpha_mask:
                    crop_filename = f"rank_{rank:04d}_sample_{sample_idx}_crop.png"
                else:
                    crop_filename = f"rank_{rank:04d}_sample_{sample_idx}_crop.jpg"
                crop_path = os.path.join(neuron_dir, crop_filename)
                save_image_with_info(cropped_image, crop_path, info)
                
                if cropped_image_without_alpha_mask is not None:
                    crop_filename = f"rank_{rank:04d}_sample_{sample_idx}_crop_without_alpha_mask.jpg"
                    crop_path = os.path.join(neuron_dir, crop_filename)
                    cropped_image_without_alpha_mask.save(crop_path)
                
                # Save activation overlay if requested (only for spatial activations)
                if args.save_overlay and resized_activation is not None:
                    overlay = create_activation_overlay(image, resized_activation)
                    overlay_filename = f"rank_{rank:04d}_sample_{sample_idx}_overlay.jpg"
                    overlay_path = os.path.join(neuron_dir, overlay_filename)
                    overlay.save(overlay_path)
                    
                    # Also save cropped overlay (always without alpha mask)
                    cropped_overlay = crop_and_resize_image(overlay, bbox, args.crop_size)
                    cropped_overlay_filename = f"rank_{rank:04d}_sample_{sample_idx}_crop_overlay.jpg"
                    cropped_overlay_path = os.path.join(neuron_dir, cropped_overlay_filename)
                    cropped_overlay.save(cropped_overlay_path)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
    
    print(f"\nProcessing complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# # Example usage for different crop_activation_regions functions:

# python /project/src/experiments/preprocessing/crop_activation_regions.py \
#     --config_file /project/src/configs/imagenet/vit_b_16_timm.yaml \
#     --indices_file /project/results/top_activations/imagenet/vit_b_16_timm/top10pct/top_activations_blocks_11_output_indices.npy \
#     --activation_file /project/results/activations/imagenet/vit_b_16_timm/activations_blocks_11_output_raw.safetensors \
#     --output_dir /project/results/cropped_regions \
#     --neuron_indices 22 \
#     --top_k_samples 100 \
#     --crop_method threshold \
#     --threshold_percentile 90 \
#     --alpha_mask

# ============================================================================
# 1. FASTEST: get_cropped_images_from_activations (when you have activations)
# ============================================================================

# Use this when you already have activations computed (like from your notebook)
# from src.experiments.preprocessing.crop_activation_regions import get_cropped_images_from_activations

# Assume you have: top_act = activations[top_indices]
# and you want to crop images for specific samples and neuron

# sample_indices = [100, 200, 300, 400, 500]
# neuron_idx = 42

# cropped_results = get_cropped_images_from_activations(
#     sample_indices=sample_indices,
#     neuron_idx=neuron_idx,
#     activations=your_activations_tensor,  # Your pre-computed activations
#     config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#     layer_name="blocks.11",
#     crop_method="threshold",
#     threshold_percentile=90.0,
#     alpha_mask=True,
#     mask_threshold=50.0
# )

# Process results
# for cropped_image, metadata in cropped_results:
#     print(f"Sample {metadata['sample_idx']}: Label {metadata['label']}, Bbox {metadata['crop_bbox']}")
#     cropped_image.save(f"cropped_sample_{metadata['sample_idx']}.png")

# ============================================================================
# 2. FAST: get_cropped_images_fast (optimized model inference)
# ============================================================================

# Use this when you need to compute activations but want better performance
# from src.experiments.preprocessing.crop_activation_regions import get_cropped_images_fast
# from models import get_fn_model_loader
# from utils.helper import load_config

# Load model
# config = load_config("/project/src/configs/imagenet/vit_b_16_timm.yaml")
# model_loader = get_fn_model_loader(config["model_name"])
# model = model_loader(config)

# cropped_results = get_cropped_images_fast(
#     sample_indices=[100, 200, 300, 400, 500],
#     neuron_idx=42,
#     model=model,
#     layer_name="blocks.11",
#     config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#     crop_method="threshold",
#     batch_size=64,  # Larger batch = faster
#     device="cuda",
#     alpha_mask=False
# )

# print(f"Processed {len(cropped_results)} samples")

# ============================================================================
# 3. CONVENIENT: get_cropped_images_from_top_activations (from indices file)
# ============================================================================

# Use this when you have pre-computed top activation indices saved
# from src.experiments.preprocessing.crop_activation_regions import get_cropped_images_from_top_activations

# cropped_results = get_cropped_images_from_top_activations(
#     indices_file="/project/results/top_activations/top10pct/top_activations_blocks_11_indices.npy",
#     neuron_idx=5,
#     model=model,
#     layer_name="blocks.11",
#     config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#     top_k_samples=10,
#     crop_method="bbox",
#     device="cuda"
# )

# print(f"Got {len(cropped_results)} cropped images for neuron 5")

# ============================================================================
# 4. BASIC: get_cropped_images (original function)
# ============================================================================

# Use this for small batches or when memory is limited
# from src.experiments.preprocessing.crop_activation_regions import get_cropped_images

# cropped_results = get_cropped_images(
#     sample_indices=[10, 20, 30],
#     neuron_idx=0,
#     model=model,
#     layer_name="blocks.11",
#     config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#     crop_method="threshold",
#     batch_size=8,  # Smaller batch for memory constraints
#     device="cuda"
# )

# ============================================================================
# 5. PATTERN MATCHING: Finding samples with specific neuron patterns
# ============================================================================

# From your notebook context: values, top = lower_simple_act.topk(k=2, dim=1)
# To find samples where specific neurons are most active:

# import torch

# Your topk results
# values, top = lower_simple_act.topk(k=2, dim=1)

# Method 1: Find exact pattern [16, 228]
# query = torch.tensor([16, 228])
# exact_matches = (top == query).all(dim=1)
# matching_sample_indices = torch.where(exact_matches)[0].tolist()

# print(f"Samples with exact pattern [16, 228]: {matching_sample_indices}")

# Method 2: Find pattern in any order
# order1 = (top == query).all(dim=1)  # [16, 228]
# order2 = (top == query.flip(0)).all(dim=1)  # [228, 16]
# any_order_matches = order1 | order2
# matching_indices_any_order = torch.where(any_order_matches)[0].tolist()

# print(f"Samples with [16, 228] in any order: {matching_indices_any_order}")

# Now get cropped images for these matching samples
# if matching_sample_indices:
#     cropped_results = get_cropped_images_from_activations(
#         sample_indices=matching_sample_indices,
#         neuron_idx=16,  # or 228, depending on which neuron you want to visualize
#         activations=lower_simple_act,
#         config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#         layer_name="lower_layer",
#         crop_method="threshold"
#     )
#     
#     print(f"Generated {len(cropped_results)} crops for pattern-matching samples")

# ============================================================================
# 6. BATCH PROCESSING: Multiple neurons at once
# ============================================================================

# Process multiple neurons efficiently
# neuron_indices = [0, 1, 2, 3, 4]
# sample_indices = list(range(100, 200))  # 100 samples

# all_results = {}
# for neuron_idx in neuron_indices:
#     print(f"Processing neuron {neuron_idx}...")
#     
#     results = get_cropped_images_from_activations(
#         sample_indices=sample_indices,
#         neuron_idx=neuron_idx,
#         activations=your_activations,
#         config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#         layer_name="blocks.11",
#         crop_method="threshold"
#     )
#     
#     all_results[neuron_idx] = results

# print(f"Processed {len(neuron_indices)} neurons for {len(sample_indices)} samples")

# ============================================================================
# 7. CREATING VISUALIZATION GRIDS
# ============================================================================

# import matplotlib.pyplot as plt

# def create_neuron_comparison_grid(activations, neuron_indices, top_k=5):
#     """Create a grid showing top activating images for multiple neurons"""
#     
#     fig, axes = plt.subplots(len(neuron_indices), top_k, 
#                             figsize=(top_k*3, len(neuron_indices)*3))
#     
#     for i, neuron_idx in enumerate(neuron_indices):
#         # Get top activating samples for this neuron
#         neuron_activations = activations[:, neuron_idx]
#         top_values, top_indices = neuron_activations.topk(k=top_k)
#         
#         # Get cropped images for top samples
#         cropped_results = get_cropped_images_from_activations(
#             sample_indices=top_indices.tolist(),
#             neuron_idx=neuron_idx,
#             activations=activations,
#             config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#             layer_name="blocks.11"
#         )
#         
#         # Plot in grid
#         for j, (cropped_image, metadata) in enumerate(cropped_results):
#             if j < top_k:
#                 axes[i, j].imshow(cropped_image)
#                 axes[i, j].set_title(f"N{neuron_idx} S{metadata['sample_idx']}\\nAct: {top_values[j]:.3f}")
#                 axes[i, j].axis('off')
#     
#     plt.tight_layout()
#     plt.savefig("neuron_comparison_grid.png", dpi=150, bbox_inches='tight')
#     plt.show()

# Usage
# create_neuron_comparison_grid(your_activations, [0, 1, 2, 3, 4])

# ============================================================================
# 8. DIFFERENT LAYER TYPES
# ============================================================================

# For CNN layers (e.g., ResNet)
# cnn_config = load_config("/project/src/configs/imagenet/resnet50.yaml")
# cnn_model = get_fn_model_loader(cnn_config["model_name"])(cnn_config)

# cnn_results = get_cropped_images_fast(
#     sample_indices=[10, 20, 30],
#     neuron_idx=100,
#     model=cnn_model,
#     layer_name="layer4.2.conv3",  # ResNet layer
#     config_file="/project/src/configs/imagenet/resnet50.yaml",
#     crop_method="threshold",
#     device="cuda"
# )

# For ViT layers with different blocks
# vit_layers = ["blocks.8", "blocks.9", "blocks.10", "blocks.11"]
# vit_results = {}

# for layer in vit_layers:
#     results = get_cropped_images_fast(
#         sample_indices=[0, 1, 2, 3, 4],
#         neuron_idx=42,
#         model=model,
#         layer_name=layer,
#         config_file="/project/src/configs/imagenet/vit_b_16_timm.yaml",
#         crop_method="threshold"
#     )
#     vit_results[layer] = results

# ============================================================================
# 9. ERROR HANDLING AND DEBUGGING
# ============================================================================

# def safe_crop_images(sample_indices, neuron_idx, activations, config_file):
#     """Wrapper with error handling and debugging"""
#     
#     try:
#         print(f"Input validation:")
#         print(f"  Activations shape: {activations.shape}")
#         print(f"  Sample indices: {len(sample_indices)} samples")
#         print(f"  Neuron index: {neuron_idx}")
#         
#         # Validate inputs
#         if neuron_idx >= activations.shape[1]:
#             raise ValueError(f"Neuron index {neuron_idx} >= {activations.shape[1]}")
#         
#         valid_samples = [idx for idx in sample_indices if idx < activations.shape[0]]
#         if len(valid_samples) != len(sample_indices):
#             print(f"Warning: {len(sample_indices) - len(valid_samples)} invalid sample indices")
#         
#         # Process
#         results = get_cropped_images_from_activations(
#             sample_indices=valid_samples,
#             neuron_idx=neuron_idx,
#             activations=activations,
#             config_file=config_file,
#             layer_name="debug_layer"
#         )
#         
#         print(f"Success: Generated {len(results)} cropped images")
#         return results
#         
#     except Exception as e:
#         print(f"Error: {e}")
#         return []

# Usage
# results = safe_crop_images([0, 1, 2], 42, your_activations, 
#                           "/project/src/configs/imagenet/vit_b_16_timm.yaml")

# ============================================================================
# 10. PERFORMANCE COMPARISON
# ============================================================================

# import time

# def benchmark_methods(sample_indices, neuron_idx, model, activations):
#     """Compare performance of different methods"""
#     
#     config_file = "/project/src/configs/imagenet/vit_b_16_timm.yaml"
#     
#     # Method 1: From activations (fastest)
#     start = time.time()
#     results1 = get_cropped_images_from_activations(
#         sample_indices, neuron_idx, activations, config_file
#     )
#     time1 = time.time() - start
#     
#     # Method 2: Fast model inference
#     start = time.time()
#     results2 = get_cropped_images_fast(
#         sample_indices, neuron_idx, model, "blocks.11", config_file, batch_size=64
#     )
#     time2 = time.time() - start
#     
#     # Method 3: Original method
#     start = time.time()
#     results3 = get_cropped_images(
#         sample_indices, neuron_idx, model, "blocks.11", config_file, batch_size=32
#     )
#     time3 = time.time() - start
#     
#     print(f"Performance comparison for {len(sample_indices)} samples:")
#     print(f"  From activations: {time1:.2f}s (fastest)")
#     print(f"  Fast inference:   {time2:.2f}s ({time2/time1:.1f}x slower)")
#     print(f"  Original method:  {time3:.2f}s ({time3/time1:.1f}x slower)")

# Usage
# benchmark_methods([0, 1, 2, 3, 4], 42, model, your_activations)