#!/bin/bash

CKPT_LIST=(
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_550.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_742.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_942.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_2.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_3.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_4.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_5.pth"
)

for CKPT_PATH in "${CKPT_LIST[@]}"
do
    # Pass the first CKPT_PATH
    if [ "$CKPT_PATH" == "${CKPT_LIST[0]}" ]; then
        echo "Processing first checkpoint: $CKPT_PATH"
    fi
    # Parse basename of CKPT_PATH
    CKPT_BASENAME=$(basename "$CKPT_PATH")

    # 1. Extract activation
    CUDA_VISIBLE_DEVICES=1 python /project/src/experiments/preprocessing/extract_activations.py \
        --layers_to_hook backbone.blocks.23 output \
        --batch_size 256 \
        --pool_type raw \
        --model_name vit_l_16_dinov3 \
        --config_file /project/src/configs/imagenet/vit_l_16_dinov3.yaml \
        --save_dir /project/results/activations/ \
        --ckpt_path "$CKPT_PATH"

    SAVED_DIR=/project/results/activations/food/vit_l_16_dinov3
    NEW_SAVE_DIR=/project/results/activations/food/vit_l_16_dinov3_$CKPT_BASENAME
    mv "$SAVED_DIR" "$NEW_SAVE_DIR"

done

# 2. Extract top index
CKPT_LIST=(
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_550.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_742.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_942.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_2.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_3.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_4.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_5.pth"
)
for CKPT_PATH in "${CKPT_LIST[@]}"
do
    CKPT_BASENAME=$(basename "$CKPT_PATH")
    ACT_SAVE_DIR=/project/results/activations/food/vit_l_16_dinov3_$CKPT_BASENAME
    SAVE_DIR=/project/results/top_activations/food/CKPT_BASENAME/top10pct

    python /project/src/experiments/preprocessing/compute_top_activations.py \
        --model_name vit_l_16_dinov3 \
        --layer_name "backbone.blocks.23" \
        --top_k 1000 \
        --aggregation top_mean \
        --top_percentile 10.0 \
        --input_file $ACT_SAVE_DIR/activations_backbone_blocks_23_output_raw.safetensors \
        --output_dir /project/results/top_activations/food/vit_l_16_dinov3_$CKPT_BASENAME/top10pct \
        --save_values
done


# 3. Extract attribution
CKPT_LIST=(
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_550.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_742.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_942.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_2.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_3.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_4.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_5.pth"
)
for CKPT_PATH in "${CKPT_LIST[@]}"
do
    CKPT_BASENAME=$(basename "$CKPT_PATH")
    TOP_SAVED_DIR=/project/results/top_activations/food/vit_l_16_dinov3_$CKPT_BASENAME/top10pct
    python /project/src/experiments/disentangling/attribution.py \
        --model_name vit_l_16_dinov3 \
        --dataset_name food \
        --tgt_layer_name "backbone.blocks.23" \
        --src_layer_name "backbone.blocks.22" \
        --all_neurons \
        --top_index_file $TOP_SAVED_DIR/top_activations_backbone_blocks_23_output_indices.npy \
        --output_dir /project/results/attributions/food/vit_l_16_dinov3_$CKPT_BASENAME
done

CKPT_LIST=(
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_550.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_742.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_step_942.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_2.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_3.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_4.pth"
    "/project/results/dinov3/Food/vit_l_slow/ckpt_epoch_5.pth"
)
for CKPT_PATH in "${CKPT_LIST[@]}"
do
    CKPT_BASENAME=$(basename "$CKPT_PATH")
    TOP_SAVED_DIR=/project/results/top_activations/food/vit_l_16_dinov3_$CKPT_BASENAME/top10pct

    SAVED_DIR=/project/results/attributions/food/vit_l_16_dinov3_$CKPT_BASENAME
    python /project/src/experiments/disentangling/greedy_clustering.py \
        --model_name vit_l_16_dinov3 \
        --tgt_layer_name "backbone.blocks.23" \
        --attr_dir $SAVED_DIR/vit_l_16_dinov3/backbone.blocks.23 \
        --top_index_file $TOP_SAVED_DIR/top_activations_backbone_blocks_23_output_indices.npy \
        --all_neurons \
        --num_samples 100 \
        --dataset_name food \
        --output_dir /project/results/clustering/food/vit_l_16_dinov3_$CKPT_BASENAME
done