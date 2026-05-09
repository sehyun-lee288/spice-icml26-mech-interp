#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Configuration
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}  # Default to 2 parallel jobs, can be overridden with environment variable
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}  # Available CUDA devices, can be overridden with environment variable


echo "Starting parallel processing with maximum $MAX_PARALLEL_JOBS concurrent jobs"
echo "Available CUDA devices: $CUDA_DEVICES"
echo "To change settings, use environment variables:"
echo "  MAX_PARALLEL_JOBS=4 CUDA_DEVICES='0,1,2,3' ./convnext.sh"

MODEL_NAME="convnext_timm" 
LAYER_LIST=(
    "stages.0.blocks.0" "stages.0.blocks.1" "stages.0.blocks.2" \
    "stages.1.blocks.0" "stages.1.blocks.1" "stages.1.blocks.2" \
    "stages.2.blocks.0" "stages.2.blocks.1" "stages.2.blocks.2" "stages.2.blocks.3" "stages.2.blocks.4" "stages.2.blocks.5" "stages.2.blocks.6" "stages.2.blocks.7" "stages.2.blocks.8" "stages.2.blocks.9" "stages.2.blocks.10" "stages.2.blocks.11" "stages.2.blocks.12" "stages.2.blocks.13" "stages.2.blocks.14" "stages.2.blocks.15" "stages.2.blocks.16" "stages.2.blocks.17" "stages.2.blocks.18" "stages.2.blocks.19" "stages.2.blocks.20" "stages.2.blocks.21" "stages.2.blocks.22" "stages.2.blocks.23" "stages.2.blocks.24" "stages.2.blocks.25" "stages.2.blocks.26" \
    "stages.3.blocks.0" "stages.3.blocks.1" "stages.3.blocks.2"
)

# echo "Step 0: Extracting raw activations for all layers..."
# python /project/src/experiments/preprocessing/extract_activations.py \
#     --layers_to_hook stages.3 output stages.2 output stages.1 output stages.0 output \
#     --batch_size 256 \
#     --pool_type raw \
#     --model_name $MODEL_NAME \
#     --config_file /project/src/configs/imagenet/$MODEL_NAME.yaml \
#     --save_dir /project/results/activations \
#     --save_intermediate

# Function to wait for jobs to complete when we reach the maximum
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 1
    done
}

# Function to get CUDA device for a job index
get_cuda_device() {
    local job_index=$1
    local devices=($(echo $CUDA_DEVICES | tr ',' ' '))
    local device_index=$((job_index % ${#devices[@]}))
    echo "${devices[$device_index]}"
}

# Function to process a single layer
process_layer() {
    local i=$1
    local job_index=$2
    
    # Get assigned CUDA device for this job
    local cuda_device=$(get_cuda_device $job_index)
    
    # --- Define layer variables for the current iteration ---
    LAYER="stages.$i"
    SAFE_LAYER="stages_$i" # For file names (e.g., blocks.11 -> blocks_11)
    
    PREV_NUM=$((i - 1))
    PREV_LAYER="stages.$PREV_NUM"
    
    echo ""
    echo "======================================================================"
    echo "Processing Target Layer: $LAYER (Source: $PREV_LAYER) [PID: $$] [CUDA: $cuda_device]"
    echo "======================================================================"

    # # --- 1. Compute Top Activating Images ---
    # # This script finds the images that most strongly activate neurons in the current LAYER.
    # echo "Step 1: Computing top activations for $LAYER..."
    # CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/preprocessing/compute_top_activations.py \
    #     --model_name $MODEL_NAME \
    #     --layer_name "$LAYER" \
    #     --input_file "/project/results/activations/imagenet/$MODEL_NAME/activations_${SAFE_LAYER}_output_raw.safetensors" \
    #     --top_k 100 \
    #     --aggregation top_mean \
    #     --top_percentile 10.0 \
    #     --output_dir "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct" \
    #     --save_values

    # --- 2. Compute Attributions ---
    # This script calculates how neurons in the PREV_LAYER contribute to the activation
    # of neurons in the current LAYER, for the top activating images found in Step 1.
    echo "Step 2: Computing attributions from $PREV_LAYER to $LAYER..."
    CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/disentangling/attribution.py \
        --model_name $MODEL_NAME \
        --tgt_layer_name "$LAYER" \
        --src_layer_name "$PREV_LAYER" \
        --all_neurons \
        --top_index_file "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct/top_activations_${SAFE_LAYER}_output_indices.npy" \
        --output_dir "/project/results/attributions/imagenet"


    # # --- 3. Run Greedy Clustering ---
    # # This script clusters the attribution maps to find "concepts" or groups of
    # # similar activation patterns from the PREV_LAYER.
    # echo "Step 3: Running greedy clustering on attributions for $LAYER on CUDA device $cuda_device..."
    # CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/disentangling/greedy_clustering.py \
    #     --model_name $MODEL_NAME \
    #     --tgt_layer_name "$LAYER" \
    #     --attr_dir "/project/results/attributions/imagenet/$MODEL_NAME/${LAYER}" \
    #     --top_index_file "/project/results/top_activations/imagenet/$MODEL_NAME/top10pct/top_activations_${SAFE_LAYER}_output_indices.npy" \
    #     --all_neurons \
    #     --threshold "0.85" \
    #     --num_samples 100 \
    #     --output_dir "/project/results/clustering/kmeans_efficient/$MODEL_NAME"
    
    echo "Completed processing layer $LAYER on CUDA device $cuda_device"
}

# Process layers in parallel with maximum limit
job_counter=0
declare -a pids=()
for i in $(seq 1 -1 1)
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
