# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from torch.profiler import profile, ProfilerActivity
import argparse




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 64  # Set your desired batch size here
MODEL_ID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jina-embeddings-v2-base-de")
# Sample German/English sentences for embedding
TEXT_SAMPLES = [
    "Künstliche Intelligenz verändert die Welt.",
    "Deep learning models require large datasets.",
    "Berlin ist die Hauptstadt von Deutschland.",
    "Transformer architectures revolutionized NLP.",
    "Die Forschung im Bereich maschinelles Lernen wächst schnell.",
]

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(SCRIPT_DIR, "traces")

# 1) Keep everything simple & fast
os.environ.pop("TORCH_LOGS", None)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model(local_only=True):
    t("loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    t("loading model")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model, tok

def make_batch(tok, bs=1):
    """
    Create a batch of tokenized text for Jina Embeddings.
    Encoder-only model — forward passes only, no generation.
    """
    texts = [TEXT_SAMPLES[i % len(TEXT_SAMPLES)] for i in range(bs)]
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    batch = {
        "input_ids": enc["input_ids"].to("cuda"),
        "attention_mask": enc["attention_mask"].to("cuda"),
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

# ----------------------------- forward pass benchmark -----------------------------
@torch.inference_mode()
def run_forward(
    model,
    batch,
    return_metrics=False,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    out = model(**batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    runtime_s = end - start
    n_seqs = int(batch["input_ids"].shape[0])
    seqs_per_s = (n_seqs / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Sequences in batch: {n_seqs}")
    t(f"🚀 Throughput: {seqs_per_s:.2f} sequences/s")

    if return_metrics:
        return {"runtime_s": runtime_s, "n_seqs": n_seqs, "throughput_seqs_s": seqs_per_s}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (default: 0 = auto-detect)')
    args = parser.parse_args()
    global BATCH_SIZE
    global TYPE
    TYPE = args.type
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    t("start")
    os.makedirs(TRACES_DIR, exist_ok=True)
    model, tok = load_model(local_only=True)
    if args.batch_size <= 0:
        import sys
        sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
        from gpu_utils import find_max_batch_size
        BATCH_SIZE = find_max_batch_size(
            make_batch_fn=lambda bs: make_batch(tok, bs=bs),
            model_fn=lambda **b: model(**b),
        )
    else:
        BATCH_SIZE = args.batch_size
    t(f"using batch_size={BATCH_SIZE}")
    batch = make_batch(tok, bs=BATCH_SIZE)

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
    t(f"benchmarking {args.runs} timed runs…")
    throughputs = []
    runtimes = []

    for i in range(args.runs):
        t(f"run {i+1}/{args.runs}…")
        m = run_forward(compiled, batch, return_metrics=True)
        throughputs.append(m["throughput_seqs_s"])
        runtimes.append(m["runtime_s"])

    # ------------------ summary stats ------------------
    avg_tps = sum(throughputs) / len(throughputs)
    p50_tps = stats.median(throughputs)
    stdev_tps = stats.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    p95_tps = sorted(throughputs)[max(0, int(round(0.95 * len(throughputs))) - 1)]
    avg_runtime = sum(runtimes) / len(runtimes)

    print("\n===== Benchmark summary =====")
    print(f"Timed runs:          {args.runs}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Avg throughput:      {avg_tps:.2f} sequences/s")
    print(f"P50 throughput:      {p50_tps:.2f} sequences/s")
    print(f"P95 throughput:      {p95_tps:.2f} sequences/s")
    print(f"Std dev (seqs/s):    {stdev_tps:.2f}")
    print(f"Avg runtime:         {avg_runtime:.3f} s")
    print(inspect.getfile(model.__class__))


    t("done")

if __name__ == "__main__":
    main()
