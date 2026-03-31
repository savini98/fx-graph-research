#!/usr/bin/env bash
# Run all new model benchmarks (original_env + fixed_env) using existing run_*.sh scripts.
# Each run_*.sh internally switches between original_env and fixed_env,
# running 30 timed runs in each to collect graph breaks, throughput, and latency.
#
# Excludes models already benchmarked: biogpt, blenderbot-400M-distill, flan-t5-large,
#   longformer-scico, phi-4-mini, qwen-audio-chat, tiny-random-PegasusForCausalLM

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    "t5-small"
    "t5-base"
    "t5-3b"
    "bart-base"
    "bart-large-cnn"
    "rebel-large"
    "opus-mt-fr-en"
    "inclusively-reformulation-it5"
    "whisper-base"
    "whisper-small"
    "whisper-large-v3"
    "MoLFormer-XL-both-10pct"
    "layoutlmv3-base"
    "grounding-dino-tiny"
    "grounding-dino-base"
    "Florence-2-large"
    "chronos-bolt-small"
)

echo "========================================="
echo " Full benchmark run for all new models"
echo " Started: $(date)"
echo "========================================="

for model in "${MODELS[@]}"; do
    model_dir="$SCRIPT_DIR/models/$model"
    run_script=$(find "$model_dir" -maxdepth 1 -name "run_*.sh" | head -1)

    if [ -z "$run_script" ]; then
        echo "⚠️  SKIP $model — no run_*.sh found"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $model ($(basename "$run_script"))"
    echo "  Time: $(date +%H:%M:%S)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    bash "$run_script" || echo "❌ $model FAILED (exit code $?)"
done

echo ""
echo "========================================="
echo " All runs completed: $(date)"
echo "========================================="
echo ""

# ==================== Summary Tables ====================

echo "==================== GRAPH BREAKS SUMMARY ===================="
printf "%-40s %-15s %-15s\n" "Model" "Original" "Fixed"
printf "%-40s %-15s %-15s\n" "-----" "--------" "-----"

for model in "${MODELS[@]}"; do
    model_dir="$SCRIPT_DIR/models/$model"

    orig_log=$(find "$model_dir/traces/" -name "*original*model_output*.log" 2>/dev/null | sort | tail -1)
    fixed_log=$(find "$model_dir/traces/" -name "*fixed*model_output*.log" 2>/dev/null | sort | tail -1)

    orig_breaks="N/A"
    fixed_breaks="N/A"

    if [ -n "$orig_log" ]; then
        b=$(grep "Graph break count:" "$orig_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$b" ] && orig_breaks="$b"
    fi
    if [ -n "$fixed_log" ]; then
        b=$(grep "Graph break count:" "$fixed_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$b" ] && fixed_breaks="$b"
    fi

    printf "%-40s %-15s %-15s\n" "$model" "$orig_breaks" "$fixed_breaks"
done

echo ""
echo "==================== THROUGHPUT SUMMARY ===================="
printf "%-40s %-20s %-20s\n" "Model" "Orig Avg Throughput" "Fixed Avg Throughput"
printf "%-40s %-20s %-20s\n" "-----" "-------------------" "--------------------"

for model in "${MODELS[@]}"; do
    model_dir="$SCRIPT_DIR/models/$model"

    orig_log=$(find "$model_dir/traces/" -name "*original*model_output*.log" 2>/dev/null | sort | tail -1)
    fixed_log=$(find "$model_dir/traces/" -name "*fixed*model_output*.log" 2>/dev/null | sort | tail -1)

    orig_tp="N/A"
    fixed_tp="N/A"

    if [ -n "$orig_log" ]; then
        t=$(grep "Avg throughput:" "$orig_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$t" ] && orig_tp="$t"
    fi
    if [ -n "$fixed_log" ]; then
        t=$(grep "Avg throughput:" "$fixed_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$t" ] && fixed_tp="$t"
    fi

    printf "%-40s %-20s %-20s\n" "$model" "$orig_tp" "$fixed_tp"
done

echo ""
echo "==================== LATENCY SUMMARY ===================="
printf "%-40s %-20s %-20s\n" "Model" "Orig Avg Runtime" "Fixed Avg Runtime"
printf "%-40s %-20s %-20s\n" "-----" "----------------" "-----------------"

for model in "${MODELS[@]}"; do
    model_dir="$SCRIPT_DIR/models/$model"

    orig_log=$(find "$model_dir/traces/" -name "*original*model_output*.log" 2>/dev/null | sort | tail -1)
    fixed_log=$(find "$model_dir/traces/" -name "*fixed*model_output*.log" 2>/dev/null | sort | tail -1)

    orig_rt="N/A"
    fixed_rt="N/A"

    if [ -n "$orig_log" ]; then
        r=$(grep "Avg runtime:" "$orig_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$r" ] && orig_rt="$r"
    fi
    if [ -n "$fixed_log" ]; then
        r=$(grep "Avg runtime:" "$fixed_log" 2>/dev/null | head -1 | sed 's/.*: //')
        [ -n "$r" ] && fixed_rt="$r"
    fi

    printf "%-40s %-20s %-20s\n" "$model" "$orig_rt" "$fixed_rt"
done
