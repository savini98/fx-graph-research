#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "layoutlmv3-base" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/microsoft/layoutlmv3-base layoutlmv3-base
fi
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

# Clean up cloned model files to free disk space
if [ -d "layoutlmv3-base" ]; then
    echo "Cleaning up cloned model files for layoutlmv3-base..."
    rm -rf "layoutlmv3-base"
    echo "Cleanup complete."
fi
