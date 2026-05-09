"""Plot DINOv3 Food-101 transfer learning statistics across training checkpoints.

Loads pre-computed training log, intra-cluster similarity stats, and cluster count
stats for a DINOv3 ViT-L backbone fine-tuned on Food-101, then generates a 3-panel
figure: (1) validation accuracy vs. steps, (2) mean intra-cluster similarity vs.
steps, (3) number of concept clusters vs. steps.

Reproduces: notebooks/n020_dinol_food.ipynb (dino_stat.pdf section) from notebooks/.

Example:
    python src/experiments/transfer/dinov3_food_analysis.py \
        --log_csv /project/dinov3/logs_slow/dino_slow_log.csv \
        --intra_sim_pkl /project/results/stats/layer_intra_sim_vit_l_16_dinov3.pkl \
        --num_clusters_json /project/results/stats/num_concept_clusters_vit_l_16_dinov3.json \
        --steps 550 742 942 1184 1776 2368 2960 \
        --output_dir /project/results/figures

Originally produces: Appx H (dino_stat.pdf)
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def main(args):
    loss_df = pd.read_csv(args.log_csv)

    with open(args.intra_sim_pkl, "rb") as f:
        intra_sim_dino = pickle.load(f)

    with open(args.num_clusters_json, "r") as f:
        num_concept_clusters_dino = json.load(f)

    x_list = args.steps

    fig = plt.figure(figsize=(12 * 0.8, 3.5 * 0.8), dpi=200)
    gs_main = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.2, 1.2])

    ax1 = fig.add_subplot(gs_main[0, 0])
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Accuracy (%)")
    ax1.tick_params(axis="x", labelsize=10)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs_main[0, 1])
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Similarity")
    ax2.tick_params(axis="x", labelsize=10)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs_main[0, 2])
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("# of Concept Clusters")
    ax3.tick_params(axis="x", labelsize=10)
    ax3.tick_params(axis="y", labelsize=10)
    ax3.grid(alpha=0.3)

    tmp_df = loss_df.dropna(subset=["val_top1"])
    ax1.plot(
        tmp_df["step"],
        tmp_df["val_top1"],
        label="Top-1 Acc.",
        marker="o",
        color="C0",
        lw=1,
    )
    ax1.plot(
        tmp_df["step"],
        tmp_df["val_top5"],
        label="Top-5 Acc.",
        marker="o",
        color="tab:red",
        lw=1,
    )
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    mean_list = [np.mean(v) for v in intra_sim_dino.values()]
    ax2.plot(
        x_list,
        mean_list,
        color="tab:blue",
        markersize=5,
        marker="^",
        linestyle="-",
        linewidth=1,
        label="ViT-L",
    )

    ax3.plot(
        x_list,
        list(num_concept_clusters_dino.values()),
        color="tab:blue",
        markersize=5,
        marker="^",
        linestyle="-",
        linewidth=1,
        label="ViT-L",
    )

    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dino_stat.pdf")
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_csv",
        type=str,
        default="/project/dinov3/logs_slow/dino_slow_log.csv",
        help="Path to DINOv3 training log CSV.",
    )
    parser.add_argument(
        "--intra_sim_pkl",
        type=str,
        default="/project/results/stats/layer_intra_sim_vit_l_16_dinov3.pkl",
        help="Pickle file with per-layer intra-cluster similarity dicts.",
    )
    parser.add_argument(
        "--num_clusters_json",
        type=str,
        default="/project/results/stats/num_concept_clusters_vit_l_16_dinov3.json",
        help="JSON file with per-layer num-concept-cluster counts.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[550, 742, 942, 1184, 1776, 2368, 2960],
        help="Training step values corresponding to keys in intra_sim and num_clusters files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/results/figures",
        help="Directory to save dino_stat.pdf.",
    )
    args = parser.parse_args()
    main(args)
