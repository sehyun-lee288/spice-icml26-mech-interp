import os
import copy
import argparse
import numpy as np
import torch
from dsets import get_imagenet, get_dataset
from safetensors.torch import load_file, save_file
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm
import time
import json
# import faiss
try:
    import cupy as cp
except ImportError:
    cp = None  # only required for >=1000-sample KDE path


from models import get_fn_model_loader
from experiments.disentangling.clustering import pairwise_cosine_similarity, batched_pairwise_cosine_similarity

def get_args():
    parser = argparse.ArgumentParser(description="Clustering")
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
        "--attr_dir",
        type=str,
        default="/project/results/attributions/imagenet/vit_b_16_timm/blocks.11"
    )
    parser.add_argument(
        "--top_index_file",
        type=str,
        default="/project/results/top_activations/imagenet/vit_b_16_timm/top10pct/top_activations_blocks_11_output_indices.npy",
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
        "--threshold",
        type=float,
        default=None,
        help="Cluster similarity threshold"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/results/clustering/kmeans/",
    )
    parser.add_argument(
        "--no_plot",
        action='store_true',
        help="Do not plot figures for a fast running"
    )
    return parser.parse_args()

def compute_similarity_threshold(attribution, save_path=None):
    if len(attribution) <= 1000:
        _, sim_mat = pairwise_cosine_similarity(attribution)
    else:
        print (f"Batched pairwise cosine similarity computation is used.")
        _, sim_mat = batched_pairwise_cosine_similarity(attribution, save_path=save_path)
        
    if isinstance(sim_mat, torch.Tensor):
        upper_tri = torch.triu(sim_mat, diagonal=1)
    elif isinstance(sim_mat, np.ndarray):
        upper_tri = np.triu(sim_mat, k=1)
    else:
        raise TypeError(f"Unexpected sim_mat type: {type(sim_mat)}")
    upper_tri_flat = upper_tri.flatten()[upper_tri.flatten() != 0.0]
    print (f"upper_tri_flat is computed. Length: {len(upper_tri_flat)}")

    # upper_tri.flatten() != 0.0] 때문에 attribution 전체가 0이면 sim_mat도 0이게 되고, upper_tri_flat = tensor([])가 됨.   
    if len(upper_tri_flat) == 0:
        return 0.0

    threshold_rank = np.quantile(upper_tri_flat, 0.95)
    
    # Find the second peak using KDE
    # 1. 전체 유사도 데이터를 사용
    similarity_values = upper_tri_flat

    # 2. KDE로 부드러운 곡선(PDF) 생성
    print (f"KDE fitting starts...")
    if len(attribution) <= 1000:
        kde = gaussian_kde(similarity_values)
        x_range = np.linspace(similarity_values.min(), similarity_values.max(), 1000)
        pdf = kde(x_range)
    else:
        def kde_cupy_hist_fft(x, bins=2048, bw=None):
            """
            Fast 1D KDE using CuPy histogram + GPU FFT convolution.
            x : CuPy array (very large)
            """

            # ---- 1. histogram on GPU ----
            xmin = float(cp.min(x))
            xmax = float(cp.max(x))

            hist, edges = cp.histogram(x, bins=bins, range=(xmin, xmax))
            hist = hist.astype(cp.float32)

            # grid centers
            grid = (edges[:-1] + edges[1:]) / 2
            dx = float(grid[1] - grid[0])

            # ---- 2. bandwidth (Scott's rule) ----
            if bw is None:
                n = x.size
                bw = 1.059 * float(cp.std(x)) * n ** (-1/5)

            # ---- 3. gaussian kernel (GPU) ----
            k_grid = (cp.arange(bins) - bins//2) * dx
            kernel = cp.exp(-0.5 * (k_grid / bw)**2)
            kernel /= kernel.sum()

            # ---- 4. FFT convolution (GPU) ----
            hist_fft = cp.fft.fft(hist)
            kernel_fft = cp.fft.fft(kernel)

            pdf = cp.real(cp.fft.ifft(hist_fft * kernel_fft))
            pdf /= (x.size * dx)

            return grid.get(), pdf.get(), bw  # return on CPU for plotting

        x_range, pdf, bw = kde_cupy_hist_fft(cp.asarray(upper_tri_flat), bins=2048)

    # 3. Scipy를 사용해 피크 찾기
    # height 파라미터로 너무 낮은 노이즈 피크는 무시할 수 있음
    peaks, _ = find_peaks(pdf, height=0.1)

    # 4. 찾은 피크들의 x좌표 값 출력
    peak_values = x_range[peaks]
    print(f"All peaks found: {peak_values}")

    # 5. 두 번째 피크 선택 (일반적으로 값이 더 큰 쪽)
    if len(peak_values) > 1:
        second_peak = peak_values[1] # 또는 np.max(peak_values)
        print(f"Second peak: {second_peak:.4f}")
    else:
        second_peak = -1000
    
    threshold = max(threshold_rank, second_peak)
    return threshold, sim_mat

def main():
    args = get_args()

    # Set neurons
    if args.all_neurons:
        from models import FEATURE_DIMS
        neuron_indices_to_process = list(range(FEATURE_DIMS[args.model_name][args.tgt_layer_name]))
        print(f"Processing all {len(neuron_indices_to_process)} neurons")
    elif args.neuron_indices is not None:
        # Process specified neurons
        neuron_indices_to_process = args.neuron_indices
        print(f"Processing {len(neuron_indices_to_process)} specified neurons: {neuron_indices_to_process}")
    else:
        # Default: process first 5 neurons
        neuron_indices_to_process = [0, 1, 2, 3, 4]
        print(f"Processing default neurons: {neuron_indices_to_process}")


    # Load model and dataset
    model = get_fn_model_loader(args.model_name)()
    model.eval().cuda();

    
    # Load dataset
    if args.dataset_name == 'imagenet':
        data_path = '/project/data/external/ILSVRC/Data/CLS-LOC'
    elif args.dataset_name == 'food':
        data_path = '/project/dinov3/data'
    else:
        raise ValueError(f"dataset_name {args.dataset_name} is not supported. It should be either 'imagenet' or 'food'")
    dataset_unnormalized = get_dataset(args.dataset_name)(
        data_path,        
        preprocessing=False,
        split="val",
        transform=None,
    )
        

    # attribution = load_file(args.attr_file)['attribution']
    top_indices = np.load(args.top_index_file) # (dim, samples)
    print (f"Top index is loaded.")

    for neuron_idx in tqdm(neuron_indices_to_process):
        # Temporary skip
        save_dir = os.path.join(args.output_dir, args.tgt_layer_name, str(args.threshold), f"{neuron_idx:04d}")
        result_file = os.path.join(save_dir, f"idx_00000_00100.pkl")
        if os.path.exists(result_file):
            continue

        indices = top_indices[neuron_idx]

        save_dir = os.path.join(args.output_dir, args.tgt_layer_name, str(args.threshold), f"{neuron_idx:04d}")
        start_idx = 0
        end_idx = args.num_samples
        tmp_save_dir = os.path.join(save_dir, f"{start_idx:05d}_{end_idx:05d}")
        os.makedirs(tmp_save_dir, exist_ok=True)

        attr_file = os.path.join(args.attr_dir, f"attribution_{neuron_idx:04d}.safetensors")
        neuron_attribution = load_file(attr_file)['attribution'] # (samples, C, H, W) OR (samples, seq_len, token_dim)
        if neuron_attribution.dim() == 3: # (samples, seq_len, token_dim)
            neuron_attribution = neuron_attribution.sum(dim=1)
        elif neuron_attribution.dim() == 4: # (samples, C, H, W)
            neuron_attribution = neuron_attribution.sum(dim=(2,3))
        # neuron_attribution will be (samples, C or token_dim)
        neuron_attribution = neuron_attribution[:args.num_samples]

        # Normalization disabled (this is the no-norm variant for Row 3 ablation).
        print (f"Neuron attribution is loaded (no normalization).")

        # Check threshold setting
        if args.threshold is None:
            if len(neuron_attribution) >= 1000:
                save_path = os.path.join(save_dir, "sim_mat.npy")
            else:
                save_path = None
            threshold, entire_sim_mat = compute_similarity_threshold(neuron_attribution, save_path)
        else:
            _, entire_sim_mat = compute_similarity_threshold(neuron_attribution, save_path)
            threshold = args.threshold
        print (f"Threshold: {threshold:.4f}")

        whole_samples = neuron_attribution.clone()
        whole_indices = indices[start_idx:end_idx].copy()
        whole_fixed_indices = list(whole_indices.copy())

        passed_groups = []
        dummy_passed_groups = None

        n_clusters = 2
        
        loop_start_time = time.time()
        loop_cnt = 0
        stuck_cnt = 0
        early_stop = 5000
        while len(whole_samples) > 0:
            # Stop condition
            if n_clusters >= whole_samples.shape[0]:
                dummy_passed_groups = copy.deepcopy(passed_groups) + [[i] for i in whole_indices]
                break

            # 1. 남은 샘플들을 n_clusters 개로 클러스터링
            print(f"Clustering {whole_samples.shape[0]} samples into {n_clusters} clusters")
            # if len(whole_samples) <= 1000:
            kmeans = KMeans(n_clusters=n_clusters, random_state=2)
            kmeans.fit(whole_samples)
            kmeans_labels = kmeans.labels_
            # else:
            #     kmeans = faiss.Kmeans(
            #             d=whole_samples.shape[1],                # embedding dimension
            #             k=n_clusters,
            #             niter=20,             # iteration 수 (보통 20~30 충분)
            #             verbose=True,
            #             gpu=True,
            #             nredo=1              # 반복 실행 횟수 (1로 두면 OK)
            #         )
            #     kmeans.train(whole_samples)
            #     distances, kmeans_labels = kmeans.index.search(whole_samples, 1)
            #     kmeans_labels = kmeans_labels.reshape(-1)

            # 2. '합격' 그룹이 있었는지 추적하는 변수
            num_passed_clusters = 0

            # 3. 각 클러스터의 유사도 검사
            loc_to_remove = []
            for i in range(n_clusters):
                cluster_idx = i
                cluster_sample_loc = np.where(kmeans_labels == cluster_idx)[0]
                cluster_original_indices = whole_indices[cluster_sample_loc]
                cluster_loc_in_ref = [whole_fixed_indices.index(i) for i in cluster_original_indices]
                # print (f"cluster_original_indices: {cluster_original_indices}")
                # print (f"cluster_loc_in_ref: {cluster_loc_in_ref}")

                sim_mat = entire_sim_mat[np.ix_(cluster_loc_in_ref, cluster_loc_in_ref)]
                if isinstance(sim_mat, torch.Tensor):
                    upper_tri = torch.triu(sim_mat, diagonal=1)
                elif isinstance(sim_mat, np.ndarray):
                    upper_tri = np.triu(sim_mat, k=1)
                else:
                    raise TypeError(f"Unexpected sim_mat type: {type(sim_mat)}")
                n, d = sim_mat.shape
                total_cnt = n * (n - 1) / 2
                if total_cnt == 0:
                    total_cnt = 1
                avg_sim = upper_tri.sum().item() / total_cnt
                # print (f"avg_sim: {avg_sim:.4f}")

                # if len(whole_samples[cluster_sample_loc]) <= 1000:
                #     avg_sim, sim_mat = pairwise_cosine_similarity(whole_samples[cluster_sample_loc])
                # else:
                #     save_path = os.path.join(save_dir, "sim_mat.npy")
                #     avg_sim, sim_mat = batched_pairwise_cosine_similarity(whole_samples[cluster_sample_loc])
                #     print (f"batched operation is used.")

                if avg_sim > threshold:
                    print(f"Cluster {cluster_idx} has average similarity {avg_sim:.4f}")
                    passed_groups.append(whole_indices[cluster_sample_loc])
                    loc_to_remove.extend(cluster_sample_loc)
                    num_passed_clusters += 1

            whole_samples = np.delete(whole_samples, loc_to_remove, axis=0)
            whole_indices = np.delete(whole_indices, loc_to_remove, axis=0)

            # 4. '합격' 그룹이 하나도 없었을 때만 클러스터 개수를 늘림
            if num_passed_clusters == 0:
                n_clusters += 1
                stuck_cnt += 1
            else:
                # 남아있는 샘플들을 클러스터링하는 optimal n_clusters가 이전 단계의 결과와 연관이 있다
                n_clusters -= (num_passed_clusters - 1)
                stuck_cnt = 0
            print('=' * 20)
            loop_cnt += 1

            if stuck_cnt >= early_stop:
                break

        # save running time
        loop_duration = time.time() - loop_start_time
        save_dir_tmp = os.path.join(args.output_dir)
        
        with open(os.path.join(save_dir_tmp, f"clustering_time_{start_idx:05d}_{end_idx:05d}.json"), "a") as f:
            log_entry = {
                "clustering_running_time": str(loop_duration),
                "clustering_iteration": loop_cnt,
                "model_name": args.model_name,
                "layer_name": args.tgt_layer_name,
                "neuron_idx": neuron_idx,
                "attr_len": len(neuron_attribution),
                "threshold": str(threshold),
            }
            f.write(json.dumps(log_entry) + "\n")

        # Merge

        if len(whole_samples) == 0: dummy_passed_groups = passed_groups
        assert sum([len(group) for group in dummy_passed_groups]) == args.num_samples

        with open(os.path.join(save_dir, f"idx_{start_idx:05d}_{end_idx:05d}.pkl"), "wb") as f:
            pickle.dump(passed_groups, f)
            
        with open(os.path.join(save_dir, f"idx_all_{start_idx:05d}_{end_idx:05d}.pkl"), "wb") as f:
            pickle.dump(dummy_passed_groups, f)

        if not args.no_plot:
            for k in range(len(passed_groups)):
                tmp_indices = passed_groups[k]

                ncols = 10
                nrows = max(1, len(tmp_indices) // ncols)

                fig, axs = plt.subplots(nrows, ncols, figsize=(10, nrows * 1.5))
                for i, ax in enumerate(axs.flat):
                    if i >= len(tmp_indices):
                        ax.axis('off')
                        continue
                    ax.imshow(dataset_unnormalized[tmp_indices[i]][0].permute(1, 2, 0))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Group {k}')
                fig.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(tmp_save_dir, f"group_{k:05d}.png"))
                plt.close()


if __name__ == "__main__":
    main()
