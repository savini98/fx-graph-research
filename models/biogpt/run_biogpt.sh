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

# Running biogpt model original
echo "Running biogpt original model..."
python biogpt_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 300 > traces/biogpt_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running biogpt original model."

echo "Running biogpt fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running biogpt model fixed
echo "Running biogpt fixed model..."
python biogpt_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 450 > traces/biogpt_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log