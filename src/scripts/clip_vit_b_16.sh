#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Configuration
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}  # Default to 2 parallel jobs, can be overridden with environment variable
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}  # Available CUDA devices, can be overridden with environment variable


echo "Starting parallel processing with maximum $MAX_PARALLEL_JOBS concurrent jobs"
echo "Available CUDA devices: $CUDA_DEVICES"
echo "To change settings, use environment variables:"
echo "  MAX_PARALLEL_JOBS=4 CUDA_DEVICES='0,1,2,3' ./vit_b_16.sh"

MODEL_NAME="clip_vit_b_16_timm"

# echo "Step 0: Extracting raw activations for all layers..."
# python /project/src/experiments/preprocessing/extract_activations.py \
#     --layers_to_hook blocks.11 output blocks.6 output blocks.2 output \
#     --batch_size 256 \
#     --pool_type raw \
#     --model_name $MODEL_NAME \
#     --config_file /project/src/configs/imagenet/$MODEL_NAME.yaml \
#     --save_dir /project/results/activations \
#     --save_intermediate

# # Function to wait for jobs to complete when we reach the maximum
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 1
    done
}

# # Function to get CUDA device for a job index
get_cuda_device() {
    local job_index=$1
    local devices=($(echo $CUDA_DEVICES | tr ',' ' '))
    local device_index=$((job_index % ${#devices[@]}))
    echo "${devices[$device_index]}"
}

# # Function to process a single layer
process_layer() {
    local i=$1
    local job_index=$2
    
    # Get assigned CUDA device for this job
    local cuda_device=$(get_cuda_device $job_index)
    
    # --- Define layer variables for the current iteration ---
    LAYER="blocks.$i"
    SAFE_LAYER="blocks_$i" # For file names (e.g., blocks.11 -> blocks_11)
    
    PREV_NUM=$((i - 1))
    PREV_LAYER="blocks.$PREV_NUM"
    
    echo ""
    echo "======================================================================"
    echo "Processing Target Layer: $LAYER (Source: $PREV_LAYER) [PID: $$] [CUDA: $cuda_device]"
    echo "======================================================================"

    # # --- 1. Compute Top Activating Images ---
    # # This script finds the images that most strongly activate neurons in the current LAYER.
    echo "Step 1: Computing top activations for $LAYER..."
    # CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/preprocessing/compute_top_activations.py \
    #     --model_name $MODEL_NAME \
    #     --layer_name "$LAYER" \
    #     --input_file "/project/results/activations/imagenet/$MODEL_NAME/activations_${SAFE_LAYER}_output_raw.safetensors" \
    #     --top_k 100 \
    #     --aggregation top_mean \
    #     --top_percentile 10.0 \
    #     --output_dir "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct" \
    #     --save_values

    # # --- 2. Compute Attributions ---
    # # This script calculates how neurons in the PREV_LAYER contribute to the activation
    # # of neurons in the current LAYER, for the top activating images found in Step 1.
    # echo "Step 2: Computing attributions from $PREV_LAYER to $LAYER..."
    # CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/disentangling/attribution.py \
    #     --model_name $MODEL_NAME \
    #     --tgt_layer_name "$LAYER" \
    #     --src_layer_name "$PREV_LAYER" \
    #     --all_neurons \
    #     --top_index_file "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct/top_activations_${SAFE_LAYER}_output_indices.npy" \
    #     --output_dir "/project/results/attributions/imagenet"


    # --- 3. Run Greedy Clustering ---
    # This script clusters the attribution maps to find "concepts" or groups of
    # similar activation patterns from the PREV_LAYER.
    echo "Step 3: Running greedy clustering on attributions for $LAYER on CUDA device $cuda_device..."
    CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/disentangling/greedy_clustering.py \
        --model_name $MODEL_NAME \
        --tgt_layer_name "$LAYER" \
        --attr_dir "/project/results/attributions/imagenet/$MODEL_NAME/${LAYER}" \
        --top_index_file "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct/top_activations_${SAFE_LAYER}_output_indices.npy" \
        --all_neurons \
        --num_samples 100 \
        --output_dir "/project/results/clustering/kmeans_efficient/$MODEL_NAME"
    
    echo "Completed processing layer $LAYER on CUDA device $cuda_device"
}

# Process layers in parallel with maximum limit
job_counter=0
declare -a pids=()
# for i in $(seq 11 -1 1)
for i in 11 6 2
do
    # Wait if we've reached the maximum number of parallel jobs
    wait_for_jobs
    
    # Start processing this layer in the background with job counter for CUDA assignment
    process_layer $i $job_counter &
    pids+=("$!")
    job_counter=$((job_counter + 1))
done

# Wait for all background jobs to complete and capture failures
failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

if [ "$failures" -ne 0 ]; then
    echo "${failures} job(s) failed. Exiting with error."
    exit 1
fi

echo ""
echo "Pipeline finished successfully."
