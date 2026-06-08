#!/usr/bin/env bash
###############################################################################
# run_g100.sh — Full, self-contained GraphMend evaluation for a rented G100 box
#
# One command runs everything on a fresh server (plain bash over SSH):
#   1. Bootstraps Miniconda if conda is not already installed
#   2. Installs git-lfs (via conda-forge, no sudo needed)
#   3. Builds the two conda envs: original_env + fixed_env (idempotent)
#   4. Pre-downloads the model weights that the per-model scripts expect locally
#   5. Runs every model's run_*.sh (original + fixed, 30 timed runs each)
#   6. Prints Graph-breaks / Throughput / Latency summary tables
#
# Usage:
#   ./run_g100.sh                  # run the full 24-model suite
#   ./run_g100.sh t5-small biogpt  # run only the named model dirs
#   REBUILD_ENVS=1 ./run_g100.sh   # force-rebuild the conda envs first
#
# Recommended on the server (survives SSH disconnect):
#   nohup ./run_g100.sh > g100_run.out 2>&1 &
#   tail -f g100_run.out
#
# Notes:
#   * Designed to be resumable/idempotent: existing conda envs and already
#     downloaded weights are reused unless REBUILD_ENVS=1.
#   * One model failing does NOT abort the rest of the suite.
#   * The full suite downloads ~100+ GB of weights and needs a CUDA GPU.
###############################################################################

# NOT -e: we want the suite to continue past a single failure.
# NOT -u: conda's activation scripts reference unbound vars and abort under set -u.
set -o pipefail

# --------------------------------------------------------------------------- #
# Configuration (override via environment, e.g. CUDA_WHEEL=cu121 ./run_g100.sh)
# --------------------------------------------------------------------------- #
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ORIGINAL_ENV="${ORIGINAL_ENV:-original_env}"
FIXED_ENV="${FIXED_ENV:-fixed_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# PyTorch CUDA wheel channel. cu118 matches the published GraphMend results.
# If the G100 has a newer GPU (e.g. Hopper sm_90) switch to cu121/cu124.
CUDA_WHEEL="${CUDA_WHEEL:-cu118}"
# Pin torch to the published GraphMend version (2.7). Newer torch (e.g. 2.12)
# changes empty-tensor reshape semantics and breaks some benchmark scripts.
# Empty TORCH_VERSION => let pip pick the channel's latest.
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
REBUILD_ENVS="${REBUILD_ENVS:-0}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

# Keep all HuggingFace downloads inside the repo so they are reused across runs.
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR" "$HF_HOME"
MASTER_LOG="$LOG_DIR/g100_full_$(date +%Y%m%d_%H%M%S).log"

# Mirror all output to the master log as well as the console.
exec > >(tee -a "$MASTER_LOG") 2>&1

# Ordered small -> large so cheap failures surface fast.
DEFAULT_MODELS=(
    "tiny-random-PegasusForCausalLM"
    "t5-small"
    "chronos-bolt-small"
    "layoutlmv3-base"
    "blenderbot-400M-distill"
    "biogpt"
    "whisper-base"
    "opus-mt-fr-en"
    "bart-base"
    "longformer-scico"
    "t5-base"
    "grounding-dino-tiny"
    "MoLFormer-XL-both-10pct"
    "whisper-small"
    "rebel-large"
    "bart-large-cnn"
    "grounding-dino-base"
    "flan-t5-large"
    "Florence-2-large"
    "inclusively-reformulation-it5"
    "whisper-large-v3"
    "t5-3b"
    "phi-4-mini"
    "qwen-audio-chat"
)

# Allow running a subset: ./run_g100.sh t5-small biogpt
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Models whose per-model run_*.sh do NOT clone weights and expect a local dir.
# Format: "<model-dir>|<huggingface-repo-url>|<destination-dir-name>"
PREDOWNLOAD=(
    "biogpt|https://huggingface.co/microsoft/biogpt|biogpt"
    "flan-t5-large|https://huggingface.co/google/flan-t5-large|flan-t5-large"
    "blenderbot-400M-distill|https://huggingface.co/facebook/blenderbot-400M-distill|blenderbot-400M-distill"
    "longformer-scico|https://huggingface.co/allenai/longformer-scico|longformer-scico"
    "phi-4-mini|https://huggingface.co/microsoft/Phi-4-mini-instruct|Phi-4-mini-instruct"
    "qwen-audio-chat|https://huggingface.co/Qwen/Qwen-Audio-Chat|Qwen-Audio-Chat"
    "tiny-random-PegasusForCausalLM|https://huggingface.co/hf-tiny-model-private/tiny-random-PegasusForCausalLM|tiny-random-PegasusForCausalLM"
)

banner() {
    echo ""
    echo "==============================================================================="
    echo " $*"
    echo "==============================================================================="
}

# --------------------------------------------------------------------------- #
# 1. Bootstrap Miniconda if conda is not on PATH
# --------------------------------------------------------------------------- #
bootstrap_conda() {
    banner "Step 1/6  Conda"
    if command -v conda >/dev/null 2>&1; then
        echo "conda already on PATH: $(command -v conda)"
    elif [ -x "$MINICONDA_DIR/bin/conda" ]; then
        echo "Found existing Miniconda at $MINICONDA_DIR"
        export PATH="$MINICONDA_DIR/bin:$PATH"
    else
        echo "Installing Miniconda into $MINICONDA_DIR ..."
        local installer="/tmp/miniconda_$(id -u).sh"
        local arch; arch="$(uname -m)"
        local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${arch}.sh"
        if command -v wget >/dev/null 2>&1; then
            wget -q "$url" -O "$installer"
        else
            curl -fsSL "$url" -o "$installer"
        fi
        bash "$installer" -b -p "$MINICONDA_DIR"
        rm -f "$installer"
        export PATH="$MINICONDA_DIR/bin:$PATH"
    fi

    # Make `conda activate` available in this non-interactive shell.
    local conda_base; conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    export PATH="$conda_base/bin:$PATH"
    echo "conda base: $conda_base"
}

# --------------------------------------------------------------------------- #
# 2. git-lfs (needed for HuggingFace weight clones)
# --------------------------------------------------------------------------- #
ensure_git_lfs() {
    banner "Step 2/6  git-lfs"
    if ! command -v git-lfs >/dev/null 2>&1; then
        echo "Installing git-lfs via conda-forge ..."
        conda install -n base -c conda-forge git-lfs -y
        export PATH="$(conda info --base)/bin:$PATH"
    fi
    git lfs install
    git-lfs version
}

# --------------------------------------------------------------------------- #
# 3. Build the two conda envs (mirrors setup_env.sh, idempotent)
# --------------------------------------------------------------------------- #
env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

create_env() {
    local name="$1" install_modified="$2"
    if env_exists "$name" && [ "$REBUILD_ENVS" != "1" ]; then
        echo ">> Env '$name' already exists — skipping (set REBUILD_ENVS=1 to rebuild)."
        return 0
    fi
    if [ "$REBUILD_ENVS" = "1" ]; then
        conda env remove -n "$name" -y >/dev/null 2>&1 || true
    fi
    echo ">> Creating env '$name' (python $PYTHON_VERSION) ..."
    conda create -n "$name" python="$PYTHON_VERSION" -y || { echo "!! conda create failed for $name"; return 1; }

    # Install via the env's own interpreter — do NOT rely on `conda activate`,
    # which can silently fall back to the base env and pollute it.
    local env_py; env_py="$(conda info --base)/envs/$name/bin/python"
    if [ ! -x "$env_py" ]; then
        echo "!! Env interpreter missing at $env_py — aborting build of '$name'."; return 1
    fi
    echo ">> Interpreter: $env_py ($("$env_py" --version 2>&1))"

    "$env_py" -m pip install --upgrade pip
    local torch_spec="torch torchvision torchaudio"
    if [ -n "$TORCH_VERSION" ]; then
        torch_spec="torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}"
    fi
    echo ">> Installing pinned torch: $torch_spec (from $CUDA_WHEEL)"
    # shellcheck disable=SC2086
    "$env_py" -m pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}" $torch_spec
    if [ "$install_modified" = "1" ]; then
        # fixed_env uses the modified transformers (the "fix").
        # NOTE: no [torch] extra — it would re-pull torch from PyPI and override
        # the CUDA_WHEEL torch we just installed, diverging the two envs.
        "$env_py" -m pip install -e "$REPO_DIR/transformers-modified"
    else
        "$env_py" -m pip install transformers==4.52.4
    fi
    "$env_py" -m pip install accelerate safetensors psutil nvidia-ml-py3 \
        matplotlib tiktoken einops transformers_stream_generator \
        sentencepiece protobuf sacremoses
    # chronos-bolt-small needs this; harmless elsewhere
    "$env_py" -m pip install chronos-forecasting || echo "  (chronos-forecasting optional install skipped)"

    "$env_py" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" \
        || { echo "!! torch import/verify failed in '$name'"; return 1; }
    "$env_py" -c "import transformers; print('transformers', transformers.__version__)"
}

# Newer conda refuses `conda create` until the channel Terms of Service are
# accepted. Accept them non-interactively (no-op on older conda without the
# `tos` subcommand).
accept_conda_tos() {
    if conda tos --help >/dev/null 2>&1; then
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
        echo "Accepted conda channel Terms of Service."
    fi
}

build_envs() {
    banner "Step 3/6  Conda environments"
    accept_conda_tos
    create_env "$ORIGINAL_ENV" 0
    create_env "$FIXED_ENV" 1
}

# --------------------------------------------------------------------------- #
# 4. Pre-download weights for models whose run_*.sh expect a local dir
# --------------------------------------------------------------------------- #
predownload_weights() {
    banner "Step 4/6  Pre-download local model weights"
    for entry in "${PREDOWNLOAD[@]}"; do
        IFS='|' read -r mdir url dest <<<"$entry"
        # Only download models that are actually scheduled to run.
        local scheduled=0
        for m in "${MODELS[@]}"; do [ "$m" = "$mdir" ] && scheduled=1; done
        [ "$scheduled" -eq 1 ] || continue

        local target="$REPO_DIR/models/$mdir/$dest"
        # Consider weights "real" only if a >1MB .bin/.safetensors exists
        # (git-LFS pointer files are a few hundred bytes — those don't count).
        if [ -d "$target" ] && find "$target" \( -name '*.safetensors' -o -name '*.bin' \) -size +1M 2>/dev/null | grep -q .; then
            echo ">> $mdir: real weights already present at $target — skipping."
            continue
        fi
        echo ">> $mdir: cloning $url -> $target"
        rm -rf "$target"
        if git clone "$url" "$target"; then
            ( cd "$target" && git lfs install >/dev/null 2>&1 && git lfs pull ) \
                || echo "   (git lfs pull reported an issue for $mdir)"
            # Strip bloat: git metadata holds a duplicate LFS object cache, and
            # HF repos ship TF/Flax copies PyTorch never uses.
            rm -rf "$target/.git"
            find "$target" \( -name '*.h5' -o -name '*.msgpack' -o -name '*.ot' \) -delete 2>/dev/null
            # If safetensors exist, the .bin is redundant (and avoids weights_only path).
            if ls "$target"/*.safetensors >/dev/null 2>&1; then
                find "$target" -name '*.bin' -delete 2>/dev/null
            fi
            local wf; wf="$(find "$target" \( -name '*.safetensors' -o -name '*.bin' \) -size +1M 2>/dev/null | head -1)"
            if [ -n "$wf" ]; then echo "   ok: $(du -sh "$target" | cut -f1) ($(basename "$wf"))"; \
            else echo "   !! $mdir still has no real weights — it will error at runtime."; fi
        else
            echo "   !! Failed to clone $mdir weights — that model will error at runtime."
        fi
    done
}

# --------------------------------------------------------------------------- #
# 5. Run every model's run_*.sh (cd into its dir so non-self-cd scripts work)
# --------------------------------------------------------------------------- #
PASSED=(); FAILED=(); SKIPPED=()

# Reclaim a model's downloaded weights once it has finished, so the disk-limited
# box can hold the next model. Keeps traces/ (the results) and scripts.
reclaim_weights() {
    local md="$1" sub
    for sub in "$md"/*/; do
        sub="${sub%/}"
        case "$(basename "$sub")" in traces|original_model_files|.jac) continue;; esac
        if find "$sub" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.h5' \
                -o -name '*.msgpack' -o -name '*.ckpt' \) 2>/dev/null | grep -q .; then
            rm -rf "$sub" && echo "   ↻ reclaimed weights: $(basename "$md")/$(basename "$sub")"
        fi
    done
    # Prune the HF hub cache too (each model uses its own entry; nothing shares).
    rm -rf "$HF_HOME/hub" 2>/dev/null
}

run_models() {
    banner "Step 5/6  Benchmark run — ${#MODELS[@]} model(s)"
    echo "Started: $(date)"
    for model in "${MODELS[@]}"; do
        local model_dir="$REPO_DIR/models/$model"
        if [ ! -d "$model_dir" ]; then
            echo "⚠️  SKIP $model — models/$model not found"; SKIPPED+=("$model"); continue
        fi
        local run_script; run_script="$(find "$model_dir" -maxdepth 1 -name 'run_*.sh' | head -1)"
        if [ -z "$run_script" ]; then
            echo "⚠️  SKIP $model — no run_*.sh found"; SKIPPED+=("$model"); continue
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ▶ $model   ($(basename "$run_script"))   $(date +%H:%M:%S)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Subshell + cd so scripts that assume cwd==their dir still work.
        if ( cd "$model_dir" && bash "$(basename "$run_script")" ); then
            echo "✅ $model done"; PASSED+=("$model")
        else
            echo "❌ $model FAILED (exit $?)"; FAILED+=("$model")
        fi
        # Free this model's weights before starting the next (disk-limited box).
        reclaim_weights "$model_dir"
    done
    echo ""
    echo "Finished: $(date)"
}

# --------------------------------------------------------------------------- #
# 6. Summary tables (reads the per-model traces/*.log files)
# --------------------------------------------------------------------------- #
print_metric_table() {
    local title="$1" grep_key="$2"
    echo ""
    echo "==================== $title ===================="
    printf "%-34s %-18s %-18s\n" "Model" "Original" "Fixed"
    printf "%-34s %-18s %-18s\n" "-----" "--------" "-----"
    for model in "${MODELS[@]}"; do
        local d="$REPO_DIR/models/$model/traces"
        local ol fl ov fv
        ol="$(find "$d" -name '*original*model_output*.log' 2>/dev/null | sort | tail -1)"
        fl="$(find "$d" -name '*fixed*model_output*.log'    2>/dev/null | sort | tail -1)"
        ov="N/A"; fv="N/A"
        [ -n "$ol" ] && ov="$(grep "$grep_key" "$ol" 2>/dev/null | head -1 | sed 's/.*: //')"
        [ -n "$fl" ] && fv="$(grep "$grep_key" "$fl" 2>/dev/null | head -1 | sed 's/.*: //')"
        printf "%-34s %-18s %-18s\n" "$model" "${ov:-N/A}" "${fv:-N/A}"
    done
}

print_summary() {
    banner "Step 6/6  Summary"
    print_metric_table "GRAPH BREAKS" "Graph break count:"
    print_metric_table "THROUGHPUT"   "Avg throughput:"
    print_metric_table "LATENCY"      "Avg runtime:"

    echo ""
    banner "Run status"
    echo "PASSED (${#PASSED[@]}): ${PASSED[*]:-none}"
    echo "FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
    echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
    echo ""
    echo "Full log: $MASTER_LOG"
}

# --------------------------------------------------------------------------- #
main() {
    banner "GraphMend G100 evaluation — $(date)"
    echo "Repo:        $REPO_DIR"
    echo "CUDA wheel:  $CUDA_WHEEL"
    echo "HF_HOME:     $HF_HOME"
    echo "Models:      ${MODELS[*]}"

    bootstrap_conda
    ensure_git_lfs
    build_envs
    predownload_weights
    run_models
    print_summary
}

main
