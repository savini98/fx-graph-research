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


# Running Qwen-Audio-Chat model original
echo "Running Qwen-Audio-Chat original model..."
python qwen_audio_chat_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 100 > traces/qwen_audio_chat_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running qwen_audio_chat original model."

echo "Running Qwen-Audio-Chat fixed model..."

# backup Qwen-Audio-Chat/modeling_qwen.py first to Qwen-Audio-Chat/modeling_qwen.py.bak
cp Qwen-Audio-Chat/modeling_qwen.py Qwen-Audio-Chat/modeling_qwen.py.bak
# in case we mess up the original file, we can restore it from here
cd Qwen-Audio-Chat/modeling_qwen.py original_model_files/modeling_qwen_original.py
# overwrite
cp fixed_model_files/modeling_qwen_fixed.py Qwen-Audio-Chat/modeling_qwen.py

# Running Qwen-Audio-Chat model fixed
echo "Running Qwen-Audio-Chat fixed model..."
python qwen_audio_chat_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 100 > traces/qwen_audio_chat_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log


# Restore the original modeling_qwen.py file
cp Qwen-Audio-Chat/modeling_qwen.py.bak Qwen-Audio-Chat/modeling_qwen.py
echo "Restored the original modeling_qwen.py file."