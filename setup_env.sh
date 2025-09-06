#!/bin/bash

# Environment name
ORIGINAL_ENV_NAME="original_env"
FIXED_ENV_NAME="fixed_env"

# Remove environments if they already exist
conda env remove -n $ORIGINAL_ENV_NAME -y >/dev/null 2>&1
conda env remove -n $FIXED_ENV_NAME -y >/dev/null 2>&1

conda create -n $ORIGINAL_ENV_NAME python=3.12 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ORIGINAL_ENV_NAME

# Install required packages
pip install --upgrade pip
# CUDA 11.8 wheels:
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
pip install transformers==4.52.4 accelerate safetensors
pip install psutil nvidia-ml-py3
pip install matplotlib tiktoken
pip install einops transformers_stream_generator
pip install sentencepiece
pip install protobuf
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"


conda create -n $FIXED_ENV_NAME python=3.12 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $FIXED_ENV_NAME

# Install required packages for the fixed environment
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
pip install -e "./transformers-modified[torch]" 
pip install accelerate safetensors
pip install psutil nvidia-ml-py3
pip install matplotlib tiktoken
pip install einops transformers_stream_generator
pip install sentencepiece
pip install protobuf
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
