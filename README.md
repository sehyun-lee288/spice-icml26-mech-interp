# SPICE: Simple Polysemantic feature Interpretation via Clustering-based Explanations

Anonymous code release for the ICML 2026 Workshop on Mechanistic Interpretability submission.

SPICE is a generalizable framework for analyzing polysemanticity in deep
vision models. It avoids architecture-dependent propagation rules,
enabling systematic comparison across CNNs and Transformers, and
automatically determines the number of concept clusters per neuron.

## Repository layout

```
src/
├── configs/imagenet/        # per-model YAML configs (ViT-B/16, ResNet-50, ConvNeXt, DenseNet, CLIP ViT, ...)
├── dsets/                   # dataset wrappers (ImageNet)
├── models/                  # model wrappers (timm ViT/ResNet, DINO)
├── scripts/                 # per-model shell entry points
├── utils/                   # helper utilities
└── experiments/
    ├── preprocessing/       # top-activating sample mining, activation extraction
    ├── running/             # SPICE main runs and ablations
    ├── disentangling/       # core SPICE algorithm (clustering, attribution, cohesion)
    ├── postprocessing/      # separability, AUC/MAD, correlation analyses
    ├── analysis/            # per-neuron / per-layer analyses
    └── transfer/            # transfer-learning / DINO probing experiments
```

## Core SPICE algorithm

The disentanglement procedure is implemented in
`src/experiments/disentangling/`:

- `attribution.py` — per-neuron attribution footprints (Input × Gradient).
- `clustering.py` / `greedy_clustering.py` — adaptive-K clustering with
  the cohesion threshold `tau = max(95th percentile of pairwise
  similarity, KDE second peak)`.
- `greedy_clustering_nonorm.py` — ablation variant without the
  per-neuron normalization step.

## Quickstart

1. Install dependencies (PyTorch, timm, transformers, scikit-learn,
   numpy, scipy, tqdm). A GPU is recommended.
2. Edit a config in `src/configs/imagenet/` to point to your local
   ImageNet `val/` directory.
3. Pick a per-model script in `src/scripts/`, e.g.:

   ```bash
   bash src/scripts/vit_b_16.sh
   ```

   The script extracts top-100 activations, computes attribution
   footprints, runs SPICE clustering, and writes per-neuron concept
   clusters.

4. Compute separability against CLIP using:

   ```bash
   python -m src.experiments.postprocessing.compute_separability_ours \
       --model vit_b_16_timm --layer blocks.11
   ```

## Notes

This codebase is shared anonymously for review only. A cleaned-up
release with full documentation will follow upon paper acceptance.

## License

Anonymous review release — please do not redistribute. A permissive
open-source license will accompany the camera-ready release.
