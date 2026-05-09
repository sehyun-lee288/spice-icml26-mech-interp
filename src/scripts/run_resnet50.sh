#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Configuration
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}  # Default to 4 parallel jobs
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}  # Available CUDA devices

# Layer configuration - update these as needed
LAYER="layer4"  # ResNet50 layer name
SAFE_LAYER="layer4"  # Safe layer name for file paths

# Range configuration
START_IDX=770
END_IDX=2047
RANGE_LIST=($(seq $START_IDX $END_IDX))
PART_SIZE=$(( (${#RANGE_LIST[@]} + 3) / 4 ))

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

# Function to process a single group
process_group() {
    local group=$1
    local job_index=$2
    
    # Get assigned CUDA device for this job
    local cuda_device=$(get_cuda_device $job_index)
    
    # Calculate sublist for this group
    local start_idx=$((group * PART_SIZE))
    local end_idx=$((start_idx + PART_SIZE - 1))
    if [ $end_idx -ge ${#RANGE_LIST[@]} ]; then
        end_idx=$((${#RANGE_LIST[@]} - 1))
    fi
    local sublist=("${RANGE_LIST[@]:$start_idx:$((end_idx - start_idx + 1))}")
    
    # Get first and last elements for display
    local first_neuron=${sublist[0]}
    local last_neuron=${sublist[$((${#sublist[@]} - 1))]}
    
    echo ""
    echo "======================================================================"
    echo "Processing Group $group (neurons $first_neuron-$last_neuron) [PID: $$] [CUDA: $cuda_device]"
    echo "======================================================================"
    echo "Neuron indices: ${sublist[@]}"

    CUDA_VISIBLE_DEVICES=$cuda_device python /project/src/experiments/disentangling/greedy_clustering.py \
        --model_name resnet50_timm \
        --tgt_layer_name "$LAYER" \
        --attr_dir "/project/results/attributions/imagenet/resnet50_timm/${LAYER}" \
        --top_index_file "/project/results/top_activations/imagenet/resnet50_timm/top10pct/top_activations_${SAFE_LAYER}_output_indices.npy" \
        --neuron_indices "${sublist[@]}" \
        --threshold "0.7" \
        --num_samples 100 \
        --output_dir "/project/results/clustering/kmeans_efficient"
    
    echo "Completed processing group $group on CUDA device $cuda_device"
}

echo "Starting parallel processing with maximum $MAX_PARALLEL_JOBS concurrent jobs"
echo "Available CUDA devices: $CUDA_DEVICES"
echo "Processing layer: $LAYER"
echo "Neuron range: $START_IDX to $END_IDX (${#RANGE_LIST[@]} neurons)"
echo "Part size: $PART_SIZE"
echo "To change settings, use environment variables:"
echo "  MAX_PARALLEL_JOBS=2 CUDA_DEVICES='0,1' ./run_resnet50.sh"

# Process groups in parallel with maximum limit
job_counter=0
for group in $(seq 0 $((PART_SIZE - 1)))
do
    # Wait if we've reached the maximum number of parallel jobs
    wait_for_jobs
    
    # Start processing this group in the background with job counter for CUDA assignment
    process_group $group $job_counter &
    job_counter=$((job_counter + 1))
done

# Wait for all background jobs to complete
wait

echo ""
echo "All groups processed successfully."

