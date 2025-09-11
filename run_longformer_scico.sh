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

# Running longformer scico model original
echo "Running longformer scico original model..."
python longformer-scico_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 300 > traces/longformer_scico_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running longformer scico original model."

echo "Running longformer scico fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running longformer scico model fixed
echo "Running longformer scico fixed model..."
python longformer-scico_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 300 > traces/longformer_scico_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log