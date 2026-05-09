"""Analyze pre-computed neuron ablation experiment results.

Loads saved ablation result .pkl files from results/ablation/<model_name>/ and
prints accuracy / mean-logit comparison across conditions (orig, random, top, bottom).

Reproduces: notebooks/n008_ablation.ipynb (accuracy comparison section) from notebooks/.

Example:
    python src/experiments/running/ablation_results_analysis.py \
        --model_name resnet50_timm \
        --result_dir /project/results/ablation

Originally supports: fig:ablations (score_vs_coverage panel analysis context)
"""

import argparse
import os
import pickle

import torch


def load_condition(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    logits = d["logits"]
    preds = (
        torch.tensor(d["preds"])
        if not isinstance(d["preds"], torch.Tensor)
        else d["preds"]
    )
    labels = (
        torch.tensor(d["labels"])
        if not isinstance(d["labels"], torch.Tensor)
        else d["labels"]
    )
    return logits, preds, labels


def analyze_model(model_name, result_dir):
    base = os.path.join(result_dir, model_name)
    conditions = {
        "orig": os.path.join(base, "orig_logits_preds_labels_random_samples.pkl"),
        "random": os.path.join(base, "random_logits_preds_labels_random_samples.pkl"),
        "top (num_cluster)": os.path.join(
            base, "num_cluster_ours_top_logits_preds_labels_random_samples.pkl"
        ),
        "bottom (num_cluster)": os.path.join(
            base, "num_cluster_ours_bottom_logits_preds_labels_random_samples.pkl"
        ),
    }

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    orig_preds = None
    for cond_name, path in conditions.items():
        if not os.path.exists(path):
            print(f"  {cond_name}: MISSING ({path})")
            continue
        logits, preds, labels = load_condition(path)
        if orig_preds is None:
            orig_preds = preds.clone()
        n = min(len(labels), len(orig_preds))
        acc = (preds[:n] == labels[:n]).float().mean().item()
        mean_logit = logits[:n][range(n), orig_preds[:n]].mean().item()
        print(f"  {cond_name:30s}  acc={acc:.4f}  mean_logit={mean_logit:.4f}")
    print(f"{'='*40}")


def main(args):
    models = (
        args.model_name
        if args.model_name
        else [
            "resnet50_timm",
            "resnet34_timm",
            "vit_b_16_timm",
            "vit_s_16_timm",
            "convnext_timm",
        ]
    )
    for m in models:
        analyze_model(m, args.result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        type=str,
        nargs="+",
        default=None,
        help="Model(s) to analyze. Defaults to all 5 backbones.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/project/results/ablation",
        help="Root directory containing per-model ablation result pkl files.",
    )
    args = parser.parse_args()
    main(args)
