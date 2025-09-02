#!/bin/bash

# Environment name
ENV_NAME="phi_env"

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y

# Create new conda environment with Python 3.12
echo "Creating conda environment with Python 3.12..."
conda create -n $ENV_NAME python=3.12 -y

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install required packages
echo "Installing PyTorch, Transformers, and other dependencies..."
pip install --upgrade pip
# CUDA 11.8 wheels:
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
pip install transformers==4.52.4 accelerate safetensors pandas
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# pip install -r requirements.txt
# conda install pytorch -c pytorch -y
# pip install transformers==4.38.1 accelerate safetensors

# Clear transformers cache to ensure clean state
echo "Clearing transformers cache..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/Phi-4-mini-instruct

git lfs install

# Remove existing clone if it exists
if [ -d "Phi-4-mini-instruct" ]; then
    echo "Removing existing Phi-4-mini-instruct directory..."
    rm -rf Phi-4-mini-instruct
fi

echo "Cloning repository..."
git clone https://huggingface.co/microsoft/Phi-4-mini-instruct

echo "Verifying LFS files..."
cd Phi-4-mini-instruct
git lfs fsck
git lfs pull
cd ..

