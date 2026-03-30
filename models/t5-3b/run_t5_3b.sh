#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "t5-3b" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/google-t5/t5-3b t5-3b
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running t5-3b model original
echo "Running t5-3b original model..."
python t5_3b_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/t5_3b_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running t5-3b original model."

echo "Running t5-3b fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running t5-3b model fixed
echo "Running t5-3b fixed model..."
python t5_3b_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/t5_3b_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "t5-3b" ]; then
    echo "Cleaning up cloned model files for t5-3b..."
    rm -rf "t5-3b"
    echo "Cleanup complete."
fi
