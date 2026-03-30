#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model locally if not already present
if [ ! -d "grounding-dino-tiny" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/IDEA-Research/grounding-dino-tiny grounding-dino-tiny
fi
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running grounding-dino-tiny model original
echo "Running grounding-dino-tiny original model..."
python grounding_dino_tiny_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/grounding_dino_tiny_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running grounding-dino-tiny original model."

echo "Running grounding-dino-tiny fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running grounding-dino-tiny model fixed
echo "Running grounding-dino-tiny fixed model..."
python grounding_dino_tiny_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/grounding_dino_tiny_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "grounding-dino-tiny" ]; then
    echo "Cleaning up cloned model files for grounding-dino-tiny..."
    rm -rf "grounding-dino-tiny"
    echo "Cleanup complete."
fi
