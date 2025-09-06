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


# Running Phi-4-mini-instruct model original
echo "Running Phi-4-mini-instruct original model..."
python phi_4_mini_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 100 > traces/phi_4_mini_instruct_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running Phi-4-mini-instruct original model."

echo "Running Phi-4-mini-instruct fixed model..."

# backup Phi-4-mini-instruct/modeling_phi3.py first to Phi-4-mini-instruct/modeling_phi3.py.bak
cp Phi-4-mini-instruct/modeling_phi3.py Phi-4-mini-instruct/modeling_phi3.py.bak
# in case we mess up the original file, we can restore it from here
cd Phi-4-mini-instruct/modeling_phi3.py original_model_files/modeling_phi3_original.py
# overwrite
cp fixed_model_files/modeling_phi3_fixed.py Phi-4-mini-instruct/modeling_phi3.py

# Running Phi-4-mini-instruct model original
echo "Running Phi-4-mini-instruct fixed model..."
python phi_4_mini_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 100 > traces/phi_4_mini_instruct_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log


# Restore the original modeling_phi3.py file
cp Phi-4-mini-instruct/modeling_phi3.py.bak Phi-4-mini-instruct/modeling_phi3.py
echo "Restored the original modeling_phi3.py file."