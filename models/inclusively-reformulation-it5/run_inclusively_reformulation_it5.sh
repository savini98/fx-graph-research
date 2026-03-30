#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model if not already present
if [ ! -d "model_weights" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/E-MIMIC/inclusively-reformulation-it5 model_weights
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running inclusively-reformulation-it5 model original
echo "Running inclusively-reformulation-it5 original model..."
python inclusively_reformulation_it5_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/inclusively_reformulation_it5_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running inclusively-reformulation-it5 original model."

echo "Running inclusively-reformulation-it5 fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running inclusively-reformulation-it5 model fixed
echo "Running inclusively-reformulation-it5 fixed model..."
python inclusively_reformulation_it5_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/inclusively_reformulation_it5_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "model_weights" ]; then
    echo "Cleaning up cloned model files for inclusively-reformulation-it5..."
    rm -rf "model_weights"
    echo "Cleanup complete."
fi
