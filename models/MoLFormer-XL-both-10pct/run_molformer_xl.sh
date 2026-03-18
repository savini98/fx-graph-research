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

# Running MoLFormer-XL-both-10pct model original
echo "Running MoLFormer-XL-both-10pct original model..."
python molformer_xl_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 256 2>&1 | tee traces/molformer_xl_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running MoLFormer-XL-both-10pct original model."

echo "Running MoLFormer-XL-both-10pct fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running MoLFormer-XL-both-10pct model fixed
echo "Running MoLFormer-XL-both-10pct fixed model..."
python molformer_xl_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 256 2>&1 | tee traces/molformer_xl_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
