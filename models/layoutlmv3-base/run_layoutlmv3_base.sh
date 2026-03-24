#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running layoutlmv3-base model original
echo "Running layoutlmv3-base original model..."
python layoutlmv3_base_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 16 2>&1 | tee traces/layoutlmv3_base_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running layoutlmv3-base original model."

echo "Running layoutlmv3-base fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running layoutlmv3-base model fixed
echo "Running layoutlmv3-base fixed model..."
python layoutlmv3_base_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 16 2>&1 | tee traces/layoutlmv3_base_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
