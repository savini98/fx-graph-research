#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

mkdir -p traces
mkdir -p original_model_files
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"


# Running flan-t5-large model original
echo "Running flan-t5-large original model..."
python flan-t5-large_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 300 > traces/flan-t5-large_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running flan-t5-large original model."

echo "Running flan-t5-large fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running flan-t5-large model fixed
echo "Running flan-t5-large fixed model..."
python flan-t5-large_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 300 > traces/flan-t5-large_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log