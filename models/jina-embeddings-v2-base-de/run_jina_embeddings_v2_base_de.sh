#!/usr/bin/env bash
# set -x
ORIGINAL="original"
FIXED="fixed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p traces
mkdir -p original_model_files

# Clone model if not already present
if [ ! -d "jina-embeddings-v2-base-de" ]; then
    echo "Cloning model..."
    git clone https://huggingface.co/jinaai/jina-embeddings-v2-base-de jina-embeddings-v2-base-de
fi

# Clone the custom code repo (jina-bert-implementation) and copy modeling files
# into the local model dir so transformers uses local code, NOT the HF cache.
if [ ! -f "jina-embeddings-v2-base-de/modeling_bert.py" ]; then
    echo "Cloning jina-bert-implementation for local custom code..."
    git clone https://huggingface.co/jinaai/jina-bert-implementation /tmp/jina-bert-impl-tmp
    cp /tmp/jina-bert-impl-tmp/modeling_bert.py jina-embeddings-v2-base-de/
    cp /tmp/jina-bert-impl-tmp/configuration_bert.py jina-embeddings-v2-base-de/
    rm -rf /tmp/jina-bert-impl-tmp

    # Patch config.json to use local files instead of jinaai/jina-bert-implementation
    python3 -c "
import json
cfg_path = 'jina-embeddings-v2-base-de/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
for k, v in cfg.get('auto_map', {}).items():
    cfg['auto_map'][k] = v.replace('jinaai/jina-bert-implementation--', '')
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patched config.json auto_map to use local files')
"
fi

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${ORIGINAL}_env"
echo "Activated conda environment"

# Running jina-embeddings-v2-base-de model original
echo "Running jina-embeddings-v2-base-de original model..."
python jina_embeddings_v2_base_de_script.py \
    --type $ORIGINAL \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/jina_embeddings_v2_base_de_original_model_output_$(date +"%Y%m%d_%H%M%S").log

echo "Completed running jina-embeddings-v2-base-de original model."

echo "Running jina-embeddings-v2-base-de fixed model..."

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${FIXED}_env"
echo "Activated conda environment"

# Running jina-embeddings-v2-base-de model fixed
echo "Running jina-embeddings-v2-base-de fixed model..."
python jina_embeddings_v2_base_de_script.py \
    --type $FIXED \
    --runs 30 \
    --batch_size 0 2>&1 | tee traces/jina_embeddings_v2_base_de_fixed_model_output_$(date +"%Y%m%d_%H%M%S").log

# Clean up cloned model files to free disk space
if [ -d "jina-embeddings-v2-base-de" ]; then
    echo "Cleaning up cloned model files for jina-embeddings-v2-base-de..."
    rm -rf "jina-embeddings-v2-base-de"
    echo "Cleanup complete."
fi
