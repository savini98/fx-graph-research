#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "bart-base" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/facebook/bart-base bart-base
fi

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running bart-base model original
echo "Running bart-base original model..."
python bart_base_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 200 2>&1 | tee traces/bart_base_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running bart-base original model."

echo "Running bart-base fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running bart-base model fixed
echo "Running bart-base fixed model..."
python bart_base_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 200 2>&1 | tee traces/bart_base_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "bart-base" ]; then
    echo "Cleaning up cloned model files for bart-base..."
    rm -rf "bart-base"
    echo "Cleanup complete."
fi
