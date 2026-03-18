#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model if not already present
if [ ! -d "rebel-large" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/Babelscape/rebel-large rebel-large
fi

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running rebel-large model original
echo "Running rebel-large original model..."
python rebel_large_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 150 2>&1 | tee traces/rebel_large_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running rebel-large original model."

echo "Running rebel-large fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running rebel-large model fixed
echo "Running rebel-large fixed model..."
python rebel_large_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 150 2>&1 | tee traces/rebel_large_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log
