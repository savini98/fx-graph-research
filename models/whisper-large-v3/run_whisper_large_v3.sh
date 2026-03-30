#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "whisper-large-v3" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/openai/whisper-large-v3 whisper-large-v3
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running whisper-large-v3 model original
echo "Running whisper-large-v3 original model..."
python whisper_large_v3_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 2 2>&1 | tee traces/whisper_large_v3_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running whisper-large-v3 original model."

echo "Running whisper-large-v3 fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running whisper-large-v3 model fixed
echo "Running whisper-large-v3 fixed model..."
python whisper_large_v3_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 2 2>&1 | tee traces/whisper_large_v3_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "whisper-large-v3" ]; then
    echo "Cleaning up cloned model files for whisper-large-v3..."
    rm -rf "whisper-large-v3"
    echo "Cleanup complete."
fi
