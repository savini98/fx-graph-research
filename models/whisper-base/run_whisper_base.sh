#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "whisper-base" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/openai/whisper-base whisper-base
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running whisper-base model original
echo "Running whisper-base original model..."
python whisper_base_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 16 2>&1 | tee traces/whisper_base_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running whisper-base original model."

echo "Running whisper-base fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running whisper-base model fixed
echo "Running whisper-base fixed model..."
python whisper_base_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 16 2>&1 | tee traces/whisper_base_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "whisper-base" ]; then
    echo "Cleaning up cloned model files for whisper-base..."
    rm -rf "whisper-base"
    echo "Cleanup complete."
fi
