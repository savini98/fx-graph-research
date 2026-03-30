#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "whisper-small" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/openai/whisper-small whisper-small
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running whisper-small model original
echo "Running whisper-small original model..."
python whisper_small_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/whisper_small_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running whisper-small original model."

echo "Running whisper-small fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running whisper-small model fixed
echo "Running whisper-small fixed model..."
python whisper_small_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/whisper_small_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "whisper-small" ]; then
    echo "Cleaning up cloned model files for whisper-small..."
    rm -rf "whisper-small"
    echo "Cleanup complete."
fi
