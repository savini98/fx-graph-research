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

# Ensure chronos-forecasting is installed
pip install chronos-forecasting 2>/dev/null

# Running chronos-bolt-small model original
echo "Running chronos-bolt-small original model..."
python chronos_bolt_small_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/chronos_bolt_small_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running chronos-bolt-small original model."

echo "Running chronos-bolt-small fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Ensure chronos-forecasting is installed
pip install chronos-forecasting 2>/dev/null

# Running chronos-bolt-small model fixed
echo "Running chronos-bolt-small fixed model..."
python chronos_bolt_small_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/chronos_bolt_small_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
