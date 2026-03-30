# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats
from datetime import datetime
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.profiler import profile, ProfilerActivity
import argparse




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 8  # Set your desired batch size here (whisper mel inputs are memory-heavy)
MODEL_ID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-small")
OUTPUT_TOKEN_LENGTH = 100  # Set your desired output token length limit here

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(SCRIPT_DIR, "traces")

# 1) Keep everything simple & fast
os.environ.pop("TORCH_LOGS", None)  # add "inductor,cudagraphs" later if you need diagnostics
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model(local_only=True):
    t("loading processor…")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    t("loading model")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model, processor

def fixed_batch(processor, bs=1):
    """
    Create a dummy mel spectrogram batch for Whisper.
    Whisper expects input_features of shape (batch, 128, 3000) for 30s audio.
    We also need decoder_input_ids for the forward pass.
    """
    import numpy as np
    # Generate random audio-like data (16kHz, 30 seconds)
    dummy_audio = np.random.randn(30 * 16000).astype(np.float32)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to("cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # Repeat for batch size
    input_features = input_features.repeat(bs, 1, 1)
    # Decoder input ids: start with the forced decoder ids (language token etc.)
    decoder_input_ids = torch.tensor([[50258]] * bs, dtype=torch.long, device="cuda")  # <|startoftranscript|>
    batch = {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
    }
    return batch

def compile_model(m):
    t("compiling with torch.compile (inductor, reduce-overhead)…")
    return torch.compile(m, backend="inductor", mode="reduce-overhead", fullgraph=False)

@torch.inference_mode()
def warmup(fn, inp, iters=1):
    t(f"warmup x{iters}…")
    for _ in range(iters):
        fn(**inp)
        torch.cuda.synchronize()

@torch.inference_mode()
def detect_cudagraphs(fn, inp, trace="trace.json"):
    t("profiling one step…")
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=False) as prof:
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
        fn(**inp)
        torch.cuda.synchronize()
    try:
        prof.export_chrome_trace(trace)
        t(f"chrome trace saved: {trace}")
    except Exception as e:
        t(f"trace export failed: {e}")

    suspects = ("cudaGraph",)
    matched = { (getattr(e,'key',None) or getattr(e,'name','')) for e in prof.key_averages()
                if any(s in (getattr(e,'key',None) or getattr(e,'name','')) for s in suspects) }
    return bool(matched), sorted(matched)

# ----------------------------- graph breaks -----------------------------
def print_graph_breaks(model, inp):
    """
    Use torch._dynamo.explain to report graph break count and reasons.
    """
    t("checking Dynamo graph breaks…")
    try:
        import torch._dynamo as dynamo
        explanation = dynamo.explain(model)(**inp)
        print(f"\n===== Graph Break Report =====")
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Number of graphs:  {explanation.graph_count}")
        if explanation.break_reasons:
            print("Break reasons:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                print(f"  {i}. {reason}")
        else:
            print("No graph breaks detected.")
        print("=" * 35)
    except Exception as e:
        t(f"graph-break analysis failed: {e}")

# ----------------------------- generation -----------------------------
@torch.inference_mode()
def generate_audio(
    model,
    processor,
    batch,
    max_new_tokens=100,
    return_metrics=False,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # For generation, pass only input_features (not decoder_input_ids)
    gen_batch = {k: v for k, v in batch.items() if k != 'decoder_input_ids'}
    out = model.generate(**gen_batch, max_new_tokens=max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    total_len = int(out.shape[1])
    new_tokens = total_len  # Whisper output is decoder-only tokens
    runtime_s = end - start
    toks_per_s = (new_tokens / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Raw new tokens generated: {new_tokens}")
    t(f"🚀 Throughput: {toks_per_s:.2f} tokens/s")

    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    if return_metrics:
        return text, {"runtime_s": runtime_s, "new_tokens": new_tokens, "throughput_tps": toks_per_s}
    return text

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size for generation (default: 0 = auto-detect)')
    args = parser.parse_args()
    global BATCH_SIZE
    global TYPE
    TYPE = args.type
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    t("start")
    os.makedirs(TRACES_DIR, exist_ok=True)
    model, processor = load_model(local_only=True)
    if args.batch_size <= 0:
        import sys
        sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
        from gpu_utils import find_max_batch_size
        BATCH_SIZE = find_max_batch_size(
            make_batch_fn=lambda bs: fixed_batch(processor, bs=bs),
            model_fn=lambda **b: model(**b),
        )
    else:
        BATCH_SIZE = args.batch_size
    t(f"using batch_size={BATCH_SIZE}")
    batch = fixed_batch(processor, bs=BATCH_SIZE)

    # Quick eager sanity check (no compile) — catches download/shape issues immediately
    t("eager forward sanity check…")
    with torch.inference_mode():
        _ = model(**batch)
        torch.cuda.synchronize()
    t("eager ok")

    # Detect and report graph breaks before compiling
    print_graph_breaks(model, batch)

    compiled = compile_model(model)
    warmup(compiled, batch, iters=1)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id = os.path.basename(MODEL_ID)
    trace_path = os.path.join(TRACES_DIR, f"{safe_model_id}_trace_{TYPE}_{dt_str}.json")
    hit, names = detect_cudagraphs(compiled, batch, trace=trace_path)
    if hit:
        t("✅ CUDA Graph activity detected:")
        for n in names: print("  -", n)
    else:
        t("❌ No CUDA Graph API events detected in this step.")

    # ------------------ benchmark runs ------------------
    import csv
    t(f"benchmarking {args.runs} timed runs…")
    throughputs = []
    runtimes = []
    new_tokens_list = []
    csv_rows = []

    for i in range(args.runs):
        t(f"run {i+1}/{args.runs}…")
        text, m = generate_audio(
            compiled, processor, batch,
            max_new_tokens=OUTPUT_TOKEN_LENGTH,
            return_metrics=True
        )
        throughputs.append(m["throughput_tps"])
        runtimes.append(m["runtime_s"])
        new_tokens_list.append(m["new_tokens"])
        csv_rows.append({
            "run": i+1,
            "throughput_tps": m["throughput_tps"],
            "runtime_s": m["runtime_s"],
            "new_tokens": m["new_tokens"]
        })

    # ------------------ summary stats ------------------
    avg_tps = sum(throughputs) / len(throughputs)
    p50_tps = stats.median(throughputs)
    stdev_tps = stats.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    p95_tps = sorted(throughputs)[max(0, int(round(0.95 * len(throughputs))) - 1)]
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_new_tokens = sum(new_tokens_list) / len(new_tokens_list)

    print("\n===== Benchmark summary =====")
    print(f"Timed runs:       {args.runs}")
    print(f"Avg throughput:   {avg_tps:.2f} tokens/s")
    print(f"P50 throughput:   {p50_tps:.2f} tokens/s")
    print(f"P95 throughput:   {p95_tps:.2f} tokens/s")
    print(f"Std dev (tps):    {stdev_tps:.2f}")
    print(f"Avg runtime:      {avg_runtime:.3f} s")
    print(f"Avg new tokens:   {avg_new_tokens:.1f}")
    print(inspect.getfile(model.__class__))


    t("done")

if __name__ == "__main__":
    main()
