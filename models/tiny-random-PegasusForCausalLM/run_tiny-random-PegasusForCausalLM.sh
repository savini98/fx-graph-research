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


# Running tiny-random-PegasusForCausalLM model original
echo "Running tiny-random-PegasusForCausalLM original model..."
python tiny-random-PegasusForCausalLM_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 150 > traces/tiny-random-PegasusForCausalLM_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running tiny-random-PegasusForCausalLM original model."

echo "Running tiny-random-PegasusForCausalLM fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running tiny-random-PegasusForCausalLM model fixed
echo "Running tiny-random-PegasusForCausalLM fixed model..."
python tiny-random-PegasusForCausalLM_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 150 > traces/tiny-random-PegasusForCausalLM_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log