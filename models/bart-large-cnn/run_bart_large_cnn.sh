#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "bart-large-cnn" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/facebook/bart-large-cnn bart-large-cnn
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running bart-large-cnn model original
echo "Running bart-large-cnn original model..."
python bart_large_cnn_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 100 2>&1 | tee traces/bart_large_cnn_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running bart-large-cnn original model."

echo "Running bart-large-cnn fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running bart-large-cnn model fixed
echo "Running bart-large-cnn fixed model..."
python bart_large_cnn_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 100 2>&1 | tee traces/bart_large_cnn_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "bart-large-cnn" ]; then
    echo "Cleaning up cloned model files for bart-large-cnn..."
    rm -rf "bart-large-cnn"
    echo "Cleanup complete."
fi
