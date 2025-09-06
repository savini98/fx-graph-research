#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

mkdir -p traces
mkdir -p original_model_files
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"


# Running blenderbot-400M-distill model original
echo "Running blenderbot-400M-distill original model..."
python blenderbot-400M-distill_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 400 > traces/blenderbot_400M_distill_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running blenderbot-400M-distill original model."

echo "Running blenderbot-400M-distill fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running blenderbot-400M-distill model fixed
echo "Running blenderbot-400M-distill fixed model..."
python blenderbot-400M-distill_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 400 > traces/blenderbot_400M_distill_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log