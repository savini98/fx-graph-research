#!/bin/bash

# set -x
TYPE="original"

# Environment name
ENV_NAME="phi_env"

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "Activated conda environment"

# Check if model files exist and have the correct size
if [ ! -f "Phi-4-mini-instruct/model-00001-of-00002.safetensors" ]; then
    echo "Error: model file not found or not properly downloaded"
    exit 1
fi

echo "line of code in modeling file: $(wc -l < Phi-4-mini-instruct/modeling_phi3.py)"
echo "original has 1180 lines, fixed has 1189 lines"

# Run the experiment
LOG_FILE="nsys_${TYPE}_output_$(date +%Y%m%d_%H%M%S).log"
PROF_FILE="nsys_${TYPE}_prof_$(date +%Y%m%d_%H%M%S)"
CSV_FILE="nsys_${TYPE}_prof_$(date +%Y%m%d_%H%M%S)_cuda_kern_exec_trace.csv"
KERNEL_STATS_FILE="kernel_stats_${TYPE}_$(date +%Y%m%d_%H%M%S).log"
# nsys profile -w true -t cuda,nvtx,cudnn,cublas \
nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --gpu-metrics-devices=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -x true -o "$PROF_FILE" \
  python experiment-nsys.py > "$LOG_FILE"

nsys stats   --report=cuda_kern_exec_trace   --format=csv   --output=. "${PROF_FILE}.nsys-rep"
python kernel_stats.py "$CSV_FILE" | tee "$KERNEL_STATS_FILE"
# set +x