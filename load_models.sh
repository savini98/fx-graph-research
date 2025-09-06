#!/usr/bin/env bash

git lfs install

# load Phi-4-mini-instruct model files from Hugging Face repository
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

# Sanity check for model file
if [ ! -f "Phi-4-mini-instruct/model-00001-of-00002.safetensors" ]; then
    echo "Error: model file not found (Phi-4-mini-instruct/model-00001-of-00002.safetensors)"
    exit 1
fi

# load Qwen-Audio-Chat model files from Hugging Face repository
if [ -d "Qwen-Audio-Chat" ]; then
    echo "Removing existing Qwen-Audio-Chat directory..."
    rm -rf Qwen-Audio-Chat
fi

echo "cloning repository"
git clone https://huggingface.co/Qwen/Qwen-Audio-Chat

echo "Verifying LFS files for qwen model..."
cd Qwen-Audio-Chat
git lfs fsck
git lfs pull
cd ..

# load blenderbot-400M-distill model files from Hugging Face repository
if [ -d "blenderbot-400M-distill" ]; then
    echo "Removing existing blenderbot-400M-distill directory..."
    rm -rf blenderbot-400M-distill
fi

echo "cloning repository"
git clone https://huggingface.co/facebook/blenderbot-400M-distill

echo "Verifying LFS files for blenderbot model..."
cd blenderbot-400M-distill
git lfs fsck
git lfs pull
cd ..

# load tiny-random-PegasusForCausalLM model files from Hugging Face repository
if [ -d "tiny-random-PegasusForCausalLM" ]; then
    echo "Removing existing tiny-random-PegasusForCausalLM directory..."
    rm -rf tiny-random-PegasusForCausalLM
fi

echo "cloning repository"
git clone https://huggingface.co/hf-tiny-model-private/tiny-random-PegasusForCausalLM

echo "Verifying LFS files for tiny-random-PegasusForCausalLM model..."
cd tiny-random-PegasusForCausalLM
git lfs fsck
git lfs pull
cd ..

# load flan-t5-large model files from Hugging Face repository
if [ -d "flan-t5-large" ]; then
    echo "Removing existing flan-t5-large directory..."
    rm -rf flan-t5-large
fi

echo "cloning repository"
git clone https://huggingface.co/google/flan-t5-large

echo "Verifying LFS files for flan-t5-large model..."
cd flan-t5-large
git lfs fsck
git lfs pull
cd ..