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

# Running opus-mt-fr-en model original
echo "Running opus-mt-fr-en original model..."
python opus_mt_fr_en_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 150 2>&1 | tee traces/opus_mt_fr_en_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running opus-mt-fr-en original model."

echo "Running opus-mt-fr-en fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running opus-mt-fr-en model fixed
echo "Running opus-mt-fr-en fixed model..."
python opus_mt_fr_en_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 150 2>&1 | tee traces/opus_mt_fr_en_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
