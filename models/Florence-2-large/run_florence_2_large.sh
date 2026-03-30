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

# Running Florence-2-large model original
echo "Running Florence-2-large original model..."
python florence_2_large_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/florence_2_large_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running Florence-2-large original model."

echo "Running Florence-2-large fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running Florence-2-large model fixed
echo "Running Florence-2-large fixed model..."
python florence_2_large_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/florence_2_large_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
