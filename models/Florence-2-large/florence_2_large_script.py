# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats, glob, shutil
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.profiler import profile, ProfilerActivity
import argparse
from PIL import Image
import numpy as np




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 4  # Set your desired batch size here (vision-language model, memory-heavy)
MODEL_ID = "microsoft/Florence-2-large"
OUTPUT_TOKEN_LENGTH = 100  # Set your desired output token length limit here
TEXT_PROMPT = "<CAPTION>"  # Florence-2 task prompt for image captioning

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(SCRIPT_DIR, "traces")

# Permanent source-of-truth copies — independent of HF cache.
ORIGINAL_MODELING_FILE = os.path.join(SCRIPT_DIR, "original_model_files/modeling_florence2.py")
FIXED_MODELING_FILE = os.path.join(SCRIPT_DIR, "../../fixed_model_files/modeling_florence2.py")

# 1) Keep everything simple & fast
os.environ.pop("TORCH_LOGS", None)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _cache_modeling_paths():
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/microsoft/Florence-2-large"
    )
    return glob.glob(os.path.join(cache_base, "*/modeling_florence2.py"))

def _clear_pyc(dst):
    pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
    for pyc in glob.glob(os.path.join(pyc_dir, "modeling_florence2*.pyc")):
        os.remove(pyc)

def save_original_if_needed():
    """
    On first run, save the HF-cached original modeling file into original_model_files/
    so we always have a clean copy to restore from.
    """
    orig_dst = os.path.abspath(ORIGINAL_MODELING_FILE)
    if os.path.exists(orig_dst):
        return  # already saved
    matches = _cache_modeling_paths()
    if matches:
        os.makedirs(os.path.dirname(orig_dst), exist_ok=True)
        shutil.copy2(matches[0], orig_dst)
        t(f"original modeling file saved → {orig_dst}")

def restore_original_modeling():
    """
    Restore the original (unpatched) modeling file from original_model_files/
    into the HF cache.
    """
    src = os.path.abspath(ORIGINAL_MODELING_FILE)
    if not os.path.exists(src):
        t("WARNING: no original modeling file found in original_model_files/ — skipping restore")
        return
    for dst in _cache_modeling_paths():
        shutil.copy2(src, dst)
        _clear_pyc(dst)
        t(f"original modeling file restored → {dst}")

def apply_fixed_modeling():
    """
    Copy the fixed modeling_florence2.py from fixed_model_files/ over the HF cache version.
    """
    src = os.path.abspath(FIXED_MODELING_FILE)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Fixed modeling file not found at {src}.")
    matches = _cache_modeling_paths()
    if not matches:
        t("WARNING: could not find cached modeling_florence2.py — skipping fix")
        return
    for dst in matches:
        shutil.copy2(src, dst)
        _clear_pyc(dst)
        t(f"fixed modeling file applied → {dst}")

def load_model(local_only=True):
    t("loading processor…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Save the original HF-cached file on first run (before any patching)
    save_original_if_needed()

    # Always restore/apply the correct modeling file before loading.
    if TYPE == "fixed":
        apply_fixed_modeling()
    else:
        restore_original_modeling()

    t("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model, processor

def make_batch(processor, bs=1):
    """
    Create a dummy image + text batch for Florence-2.
    Florence-2 takes an image and a task prompt (e.g. <CAPTION>, <OD>, etc.)
    """
    # Create a dummy RGB image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    # Process single example first
    inputs = processor(text=TEXT_PROMPT, images=dummy_image, return_tensors="pt")

    # Move to GPU and repeat for batch
    # Cast float tensors to float16 to match model dtype (avoids conv2d dtype mismatch)
    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    batch = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                v = v.to("cuda", dtype=target_dtype)
            else:
                v = v.to("cuda")
            # Repeat for batch size
            if v.dim() >= 1:
                repeat_dims = [bs] + [1] * (v.dim() - 1)
                v = v.repeat(*repeat_dims)
            batch[k] = v
        else:
            batch[k] = v

    # Florence-2's BART-like language model requires decoder_input_ids for the forward pass.
    # Use the BOS token (decoder_start_token_id=2 for BART) as the start token.
    if "decoder_input_ids" not in batch and "input_ids" in batch:
        bos_id = 2  # BART decoder_start_token_id
        decoder_input_ids = torch.full(
            (batch["input_ids"].shape[0], 1), bos_id,
            dtype=batch["input_ids"].dtype, device=batch["input_ids"].device
        )
        batch["decoder_input_ids"] = decoder_input_ids

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
def generate_text(
    model,
    processor,
    batch,
    max_new_tokens=100,
    return_metrics=False,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    out = model.generate(
        input_ids=batch["input_ids"],
        pixel_values=batch.get("pixel_values"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    total_len = int(out.shape[1])
    new_tokens = total_len
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
    model, processor = load_model(local_only=True)
    if args.batch_size <= 0:
        import sys
        sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
        from gpu_utils import find_max_batch_size
        BATCH_SIZE = find_max_batch_size(
            make_batch_fn=lambda bs: make_batch(processor, bs=bs),
            model_fn=lambda **b: model(**b),
        )
    else:
        BATCH_SIZE = args.batch_size
    t(f"using batch_size={BATCH_SIZE}")
    batch = make_batch(processor, bs=BATCH_SIZE)

    # Quick eager sanity check (no compile) — catches download/shape issues immediately
    t("eager forward sanity check…")
    with torch.inference_mode():
        _ = model(**batch)
        torch.cuda.synchronize()
    t("eager ok")

    # Detect and report graph breaks before compiling
    print_graph_breaks(model, batch)

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from profiling_utils import profile_small_batch
    profile_small_batch(model, batch, compile_model,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", f"profile_{TYPE}.json"))
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
    new_tokens_list = []

    for i in range(args.runs):
        t(f"run {i+1}/{args.runs}…")
        text, m = generate_text(
            compiled, processor, batch,
            max_new_tokens=OUTPUT_TOKEN_LENGTH,
            return_metrics=True
        )
        throughputs.append(m["throughput_tps"])
        runtimes.append(m["runtime_s"])
        new_tokens_list.append(m["new_tokens"])

    # ------------------ summary stats ------------------
    avg_tps = sum(throughputs) / len(throughputs)
    p50_tps = stats.median(throughputs)
    stdev_tps = stats.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    p95_tps = sorted(throughputs)[max(0, int(round(0.95 * len(throughputs))) - 1)]
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_new_tokens = sum(new_tokens_list) / len(new_tokens_list)

    print("\n===== Benchmark summary =====")
    print(f"Timed runs:       {args.runs}")
    print(f"Batch size:       {BATCH_SIZE}")
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
