# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats, glob, shutil
from datetime import datetime
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.profiler import profile, ProfilerActivity
import argparse

# Permanent source-of-truth copies stored in this repo — independent of HF cache.
# original_model_files/ is committed to git so it survives cache clears and re-clones.
ORIGINAL_MODELING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "original_model_files/modeling_molformer.py"
)
FIXED_MODELING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../fixed_model_files/modeling_molformer.py"
)

def _cache_modeling_paths():
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/"
        "ibm-research/MoLFormer-XL-both-10pct"
    )
    return glob.glob(os.path.join(cache_base, "*/modeling_molformer.py"))

def _clear_pyc(dst):
    pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
    for pyc in glob.glob(os.path.join(pyc_dir, "modeling_molformer*.pyc")):
        os.remove(pyc)

def restore_original_modeling():
    """
    Restore the original (unpatched) modeling file from original_model_files/
    into the HF cache. This is the permanent source of truth — it will work
    correctly regardless of run order or whether the HF cache was cleared.
    """
    src = os.path.abspath(ORIGINAL_MODELING_FILE)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Original modeling file not found at {src}. "
            "Re-clone the model and copy modeling_molformer.py into original_model_files/."
        )
    matches = _cache_modeling_paths()
    if not matches:
        t("WARNING: could not find cached modeling_molformer.py — model may not be downloaded yet")
        return
    for dst in matches:
        shutil.copy2(src, dst)
        _clear_pyc(dst)
        t(f"original modeling file restored → {dst}")

def apply_fixed_modeling():
    """
    Copy the fixed modeling_molformer.py from fixed_model_files/ over the HF cache version.
    """
    src = os.path.abspath(FIXED_MODELING_FILE)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Fixed modeling file not found at {src}.")
    matches = _cache_modeling_paths()
    if not matches:
        t("WARNING: could not find cached modeling_molformer.py — skipping fix")
        return
    for dst in matches:
        shutil.copy2(src, dst)
        _clear_pyc(dst)
        t(f"fixed modeling file applied → {dst}")




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 256  # Set your desired batch size here
MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
# Representative SMILES strings for benchmarking
SMILES_SAMPLES = [
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2ccccc2c1",                # Naphthalene
    "CCO",                           # Ethanol
]

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
    t("loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Always restore/apply the correct modeling file before loading.
    # Source of truth is original_model_files/ (committed to git) — not the HF cache.
    # This guarantees correct behaviour regardless of run order or cache state.
    if TYPE == "fixed":
        apply_fixed_modeling()
    else:
        restore_original_modeling()

    t("loading model")
    # Fix 1: set deterministic_eval=True so orthogonal_random_weights() is NOT
    # called on every forward pass (only during __init__). Without this, the
    # buffer is re-registered each step, which is incompatible with CUDA Graphs.
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.deterministic_eval = True

    # MoLFormer must stay in float32: geqrf (QR decomp) is not implemented for
    # float16 on CPU, and the orthogonal weight buffer is not converted by .half().
    model = AutoModel.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tok

def make_batch(tok, bs=1, include_attention_mask=True):
    """
    Create a batch of tokenized SMILES strings for MoLFormer.
    MoLFormer is encoder-only — we do forward passes, not generation.

    include_attention_mask=True  → original run: triggers torch.equal() graph breaks
    include_attention_mask=False → fixed run: omits mask, skipping the graph-break branch
                                   (safe for uniform-length SMILES benchmarking)
    """
    smiles_batch = [SMILES_SAMPLES[i % len(SMILES_SAMPLES)] for i in range(bs)]
    enc = tok(
        smiles_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    batch = {"input_ids": enc["input_ids"].to("cuda")}
    if include_attention_mask:
        batch["attention_mask"] = enc["attention_mask"].to("cuda")
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
    # For encoder models, report sequences/second as throughput
    n_seqs = int(batch["input_ids"].shape[0])
    toks_per_s = (n_seqs / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Sequences in batch: {n_seqs}")
    t(f"🚀 Throughput: {toks_per_s:.2f} sequences/s")

    if return_metrics:
        return {"runtime_s": runtime_s, "n_seqs": n_seqs, "throughput_seqs_s": toks_per_s}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    args = parser.parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    global TYPE
    TYPE = args.type
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    t("start")
    os.makedirs(TRACES_DIR, exist_ok=True)
    model, tok = load_model(local_only=True)
    # Original run: include attention_mask to reproduce the torch.equal() graph breaks.
    # Fixed run: omit attention_mask (the patched model eliminates the graph-break branch).
    batch = make_batch(tok, bs=BATCH_SIZE, include_attention_mask=(TYPE == "original"))

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
    print(inspect.getfile(compiled.__class__))


    t("done")

if __name__ == "__main__":
    main()
