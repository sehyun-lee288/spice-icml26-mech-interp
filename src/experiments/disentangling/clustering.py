import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Sequence, Any, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import logging
from experiments.preprocessing.compute_top_activations import aggregate_spatial_dimensions
import torch.nn.functional as F
# import faiss
import os

def batched_pairwise_cosine_similarity(x, save_path, chunk_size=5000, return_matrix=True):
    """
    Batched pairwise cosine similarity using Faiss (CPU/GPU supported).
    X: numpy array, shape [n, d], float32
    """
    if os.path.exists(save_path):
        try:
            sim = np.memmap(save_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(5000, 5000))
            sim_mat = np.array(sim)
            n = sim_mat.shape[0]
            # sum 전체 - 대각선 합 / n*(n-1)
            mean_no_diag = (sim_mat.sum() - np.trace(sim_mat)) / (n*(n-1))
            print("Mean similarity (excluding diagonal):", mean_no_diag)
            return mean_no_diag, sim_mat
        except:
            print (f"File {save_path} exists, but unable to load. Recompute.")
            pass

    # Check x.dtype and convert to np.float32 if necessary
    if x.dtype != np.float32:
        print(f"Input x is not float32, but {x.dtype}. Converting to float32 for faiss computations.")
        x = x.cpu().numpy().astype(np.float32)

    # assert x.dtype == np.float32, "Faiss requires float32."

    n, d = x.shape

    # -------------------------
    # Normalize (cosine sim)
    # -------------------------
    X_norm = x.copy()
    faiss.normalize_L2(X_norm)

    # -------------------------
    # Prepare output
    # -------------------------
    sim_mat = None
    if return_matrix:
        sim_mat = np.memmap(save_path,
                            dtype=np.float32,
                            mode="w+",
                            shape=(n, n))

    # -------------------------
    # Faiss IP index
    # -------------------------
    # GPU 사용 가능
    use_gpu = False

    if use_gpu:
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = faiss.IndexFlatIP(d)

    # 데이터 전체 추가
    index.add(X_norm)

    # 전체 sum 계산용
    total_sum = 0.0
    total_cnt = n * (n - 1) / 2

    # -------------------------
    # Batched similarity
    # -------------------------
    for i in tqdm(range(0, n, chunk_size), desc="Outer chunks"):
        end_i = min(i + chunk_size, n)
        Xi = X_norm[i:end_i]   # shape [b1, d]

        # Xi 와 전체에 대한 similarity
        D_total, _ = index.search(Xi, n)   # [b1, n]

        # chunk 단위로 저장
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            sim_chunk = D_total[:, j:end_j]   # [b1, b2]

            # 저장
            if return_matrix:
                sim_mat[i:end_i, j:end_j] = sim_chunk

            # sum (upper triangular)
            if j < i:
                continue  # 아래 삼각형은 나중에 i > j 에서 처리됨
            elif i == j:
                # upper triangular only
                mask = np.triu(np.ones_like(sim_chunk), k=1).astype(bool)
                total_sum += sim_chunk[mask].sum()
            else:
                # full sum (we're in upper triangle)
                total_sum += sim_chunk.sum()

    mean_sim = total_sum / total_cnt

    return mean_sim, sim_mat


def pairwise_cosine_similarity(x):
    """
    x: [num_samples, hidden_dim]
    return: [num_samples, num_samples]
    """
    
    sim_mat = torch.nn.functional.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
    
    # Take the upper triangle of the matrix
    upper_tri = torch.triu(sim_mat, diagonal=1)
    
    # Take the mean of the upper triangle
    cnt = x.shape[0] * (x.shape[0] - 1) / 2
    return upper_tri.sum() / cnt, sim_mat

class InputGradientClusterer:
    """
    Computes Input * Gradient for specified source layers with respect to target neurons
    and performs clustering to disentangle features.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 save_dir: Union[str, Path] = "results/clustering",
                 device: Optional[torch.device] = None):
        """
        Initialize the InputGradientClusterer.
        
        Args:
            model: The PyTorch model to analyze
            save_dir: Directory to save results and images
            device: Device to run computations on
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_layer_by_name(self, layer_name: str, model: nn.Module) -> nn.Module:
        """Get layer object by name, supporting nested paths like 'encoder.blocks.0'"""
        parts = layer_name.split('.')
        layer = model
        
        try:
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    if hasattr(layer, '__getitem__'):
                        layer = layer[idx]
                    else:
                        layer = getattr(layer, part)
                else:
                    layer = getattr(layer, part)
            return layer
        except (AttributeError, IndexError, TypeError) as e:
            raise AttributeError(f"Layer '{layer_name}' not found in model: {e}")
    
    def compute_input_gradient(self,
                             input_tensor: torch.Tensor,
                             tgt_layer_name: str,
                             src_layer_names: List[Tuple[str, str]],
                             tgt_neurons: Optional[Union[int, slice, Sequence[int]]] = None
                             ) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        Computes Input * Gradient for specified source layers with respect to target neurons.

        Args:
            input_tensor: A single input tensor for the model (e.g., shape [1, C, H, W])
            tgt_layer_name: The string name of the target layer
            src_layer_names: A list of tuples (layer_name, target_type) for source layers
            tgt_neurons: Specifies the target neurons (channels) in the target layer

        Returns:
            Dictionary where keys are (layer_name, target_type) tuples and
            values are the corresponding Input * Gradient attribution maps
        """
        self.model.eval()
        
        source_activations = {}
        source_gradients = {}
        target_activation = {}
        hooks = []

        def forward_hook_fn(name: str, target_type: str = 'output'):
            def hook(module: nn.Module, input: Any, output: torch.Tensor):
                tensor_to_save = None
                
                if target_type == 'output':
                    if isinstance(output, torch.Tensor):
                        tensor_to_save = output
                    elif isinstance(output, (tuple, list)):
                        tensor_to_save = output[0]
                elif target_type == 'input':
                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        tensor_to_save = input[0]

                if tensor_to_save is not None:
                    source_activations[(name, target_type)] = tensor_to_save
                    tensor_to_save.requires_grad_()
                    tensor_to_save.register_hook(
                        lambda grad: source_gradients.update({(name, target_type): grad})
                    )
            return hook

        def target_forward_hook_fn(module: nn.Module, inp: Any, out: torch.Tensor):
            target_activation['output'] = out

        # Register hooks on source layers
        for name, target_type in src_layer_names:
            try:
                module = self._get_layer_by_name(name, self.model)
                handle = module.register_forward_hook(forward_hook_fn(name, target_type))
                hooks.append(handle)
            except AttributeError as e:
                self.logger.error(f"Source layer '{name}' not found: {e}")
                raise

        # Register hook on target layer
        try:
            target_module = self._get_layer_by_name(tgt_layer_name, self.model)
            handle = target_module.register_forward_hook(target_forward_hook_fn)
            hooks.append(handle)
        except AttributeError as e:
            self.logger.error(f"Target layer '{tgt_layer_name}' not found: {e}")
            raise

        # Forward pass
        input_tensor = input_tensor.to(self.device)
        self.model(input_tensor)

        if 'output' not in target_activation:
            raise RuntimeError(f"Failed to capture activation from target layer '{tgt_layer_name}'")

        target_act = target_activation['output']

        # Select target neurons and aggregate
        if tgt_neurons is not None:
            selected_target = target_act[:, tgt_neurons]
        else:
            selected_target = target_act

        target_scalar = selected_target.sum()

        # Backward pass
        self.model.zero_grad()
        target_scalar.backward()

        # Clean up hooks
        for handle in hooks:
            handle.remove()

        # Compute attributions
        attributions = {}
        for name, target_type in src_layer_names:
            if (name, target_type) in source_activations and (name, target_type) in source_gradients:
                act = source_activations[(name, target_type)]
                grad = source_gradients[(name, target_type)]
                attributions[(name, target_type)] = act * grad
            else:
                self.logger.warning(f"Activation or gradient for '{name}' not captured")

        return attributions
    
    def perform_recursive_clustgering(self,
                                      attributions: Dict[Tuple[str, str], torch.Tensor],
                                      tgt_layer_name: str,
                                      neuron_idx: int,
                                      flatten_method: str = "top_mean",
                                      random_state: int = 42) -> Dict[str, Any]:
        """
        Support only one layer
        """
        clustering_results = {}
        
        # HARD CODE
        from dsets import get_imagenet
        
        dataset_unnormalized = get_imagenet(
            data_path='/project/data/external/ILSVRC/Data/CLS-LOC',
            preprocessing=False,
            split='val'
        )
                
        from safetensors import safe_open
        with safe_open("/project/results/top_activations/top10pct/blocks_11_output_values.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                vec = f.get_tensor(key).cpu()
        
        with safe_open("/project/results/top_activations/top10pct/blocks_10_output_values.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                lower_vec = f.get_tensor(key).cpu()
        
        values, indices = vec[:, neuron_idx].sort(descending=True)
        
        sub_indices = indices[:1000].clone()
        
        # Recursively clustering less similar samples if similarity is less than 0.95
        whole_samples = lower_vec[sub_indices].clone()
        indices = sub_indices.clone()
        passed_groups = []

        n_clusters = 1
        while True:
            # Stop condition
            if whole_samples.shape[0] <= n_clusters:
                break
            
            # Cluster the samples
            print (f"Clustering {whole_samples.shape[0]} samples into {n_clusters} clusters")
            kmeans = KMeans(n_clusters=n_clusters, random_state=2)
            kmeans.fit(whole_samples)
            kmeans_labels = kmeans.labels_

            loc_to_remove = []
            for i in range(n_clusters):
                cluster_idx = i
                cluster_sample_loc = np.where(kmeans_labels == cluster_idx)[0]
                avg_sim, sim_mat = pairwise_cosine_similarity(whole_samples[cluster_sample_loc])
                
                if avg_sim > 0.95:
                    print (f"Cluster {cluster_idx} has average similarity {avg_sim:.4f}")
                    passed_groups.append(indices[cluster_sample_loc])
                    loc_to_remove.extend(cluster_sample_loc)

            whole_samples = np.delete(whole_samples, loc_to_remove, axis=0)
            indices = np.delete(indices, loc_to_remove, axis=0)
            
            n_clusters += 1
            print ('='*20)

        # Save results
        import os
        save_dir = f"/project/results/clustering/kmeans/blocks.11/n_{neuron_idx:003d}"
        os.makedirs(save_dir)
        
        np.save(os.path.join(save_dir, "idx_0_1000"), np.array(passed_groups))
        
        for k in range(len(passed_groups)):
            indices = passed_groups[k]

            ncols = 10
            nrows = max(1, len(indices) // ncols)

            fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows*1.5))
            for i, ax in enumerate(axs.flat):
                if i >= len(indices):
                    ax.axis('off')
                    continue
                ax.imshow(dataset_unnormalized[indices[i]][0].permute(1,2,0))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Group {k}')
            fig.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(save_dir, f"group_{k:05d}.png"))
    
    def perform_clustering(self,
                          attributions: Dict[Tuple[str, str], torch.Tensor],
                          num_clusters: int = 10,
                          flatten_method: str = "top_mean",
                          random_state: int = 42) -> Dict[str, Any]:
        """
        Perform clustering on attribution maps to disentangle features.
        
        Args:
            attributions: Attribution maps from compute_input_gradient
            num_clusters: Number of clusters for k-means
            flatten_method: Method to flatten spatial dimensions ('global_avg_pool', 'flatten', 'spatial_mean')
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing clustering results
        """
        clustering_results = {}
        
        for (layer_name, target_type), attribution in attributions.items():
            self.logger.info(f"Clustering {layer_name} ({target_type}) with shape {attribution.shape}")
            
            
            # Flatten spatial dimensions
            if len(attribution.shape) == 3:  # [B, seq_len, C]
                if flatten_method == "top_mean":
                    flattened = aggregate_spatial_dimensions(
                        attribution,
                        aggregation='top_mean',
                        top_percentile=10.0,
                        type='vit'
                    )
                else:
                    raise ValueError(f"Unknown flatten_method: {flatten_method}")
            elif len(attribution.shape) == 4:  # [B, C, H, W]
                if flatten_method == "top_mean":
                    flattened = aggregate_spatial_dimensions(
                        attribution,
                        aggregation='top_mean',
                        top_percentile=10.0,
                        type='vit'
                    )
                else:
                    raise ValueError(f"Unknown flatten_method: {flatten_method}")
            else:
                raise ValueError(f"Unknown attribution shape: {attribution.shape}")
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(flattened)
            
            # Store results
            layer_results = {
                'cluster_labels': cluster_labels,
                'flattened_activations': flattened,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'n_iter': kmeans.n_iter_,
                'attribution_shape': attribution.shape,
                'flattened_shape': flattened.shape,
                'flatten_method': flatten_method,
                'num_clusters': num_clusters
            }
            
            clustering_results[f"{layer_name}_{target_type}"] = layer_results
            
        return clustering_results
    
    def save_results(self, 
                    clustering_results: Dict[str, Any],
                    attributions: Dict[Tuple[str, str], torch.Tensor],
                    metadata: Dict[str, Any],
                    experiment_name: str = None) -> str:
        """
        Save clustering results, attributions, and metadata.
        
        Args:
            clustering_results: Results from perform_clustering
            attributions: Original attribution maps
            metadata: Additional metadata to save
            experiment_name: Name for the experiment (auto-generated if None)
            
        Returns:
            Path to the saved experiment directory
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"clustering_experiment_{timestamp}"
        
        exp_dir = self.save_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clustering results
        with open(exp_dir / "clustering_results.pkl", 'wb') as f:
            pickle.dump(clustering_results, f)
        
        # Save attributions
        attributions_cpu = {}
        for key, value in attributions.items():
            attributions_cpu[f"{key[0]}_{key[1]}"] = value.detach().cpu().numpy()
        
        with open(exp_dir / "attributions.pkl", 'wb') as f:
            pickle.dump(attributions_cpu, f)
        
        # Save metadata
        metadata['experiment_name'] = experiment_name
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['device'] = str(self.device)
        
        with open(exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Results saved to {exp_dir}")
        return str(exp_dir)
    
    def plot_clustering_results(self,
                               cluster_labels: np.ndarray,
                               img_list: List[torch.Tensor],
                               save_dir: Union[str, Path] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create visualization plots for clustering results.
        Plot images in the same order as the clustering results.
        
        Args:
            clustering_results: Results from perform_clustering
            img_list: List of images to plot
            save_path: Path to save the plot (if None, uses self.save_dir)
            figsize: Figure size for the plot
        """
        if save_dir is None:
            save_dir = self.save_dir

        for k in range(len(np.unique(cluster_labels))):
            cluster_sample_loc = np.where(cluster_labels == k)[0]

            ncols = 10
            nrows = max(1, len(cluster_sample_loc) // ncols)

            fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows*1.5))
            for i, ax in enumerate(axs.flat):
                if i >= len(cluster_sample_loc):
                    ax.axis('off')
                    continue
                ax.imshow(img_list[cluster_sample_loc[i]].permute(1,2,0))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Cluster {k}')
            fig.tight_layout()
            # plt.show()
            save_path = save_dir / f"clustering_visualization_{k:03d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Clustering visualization saved to {save_path}")

        
    def plot_similarity_matrix(self,
                               cluster_labels: np.ndarray,
                               flattened_activations: np.ndarray,
                               save_dir: Union[str, Path] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create similarity matrix plot for clustering results.
        """
        num_clusters = len(np.unique(cluster_labels))
        centroid = torch.zeros(num_clusters, flattened_activations.shape[1])
        similarity_list = []
        for k in range(num_clusters):
            cluster_sample_loc = np.where(cluster_labels == k)[0]
            centroid[k] = flattened_activations[cluster_sample_loc].mean(dim=0)
            similarity, _ = pairwise_cosine_similarity(flattened_activations[cluster_sample_loc])
            similarity_list.append(similarity)
        
        similarity, sim_mat = pairwise_cosine_similarity(centroid)
        for k in range(num_clusters):
            sim_mat[k, k] = similarity_list[k]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_mat, cmap='viridis', vmin=0, vmax=1, ax=ax, annot=True, fmt='.2f')
        ax.set_title('Cosine Similarity Matrix between clusters')

        save_path = save_dir / f"similarity_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()