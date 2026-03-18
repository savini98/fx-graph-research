# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torch.profiler import profile, ProfilerActivity
import argparse
from PIL import Image
import numpy as np




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 8  # Set your desired batch size here (vision models are memory-heavy)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "a cat. a dog. a person."

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
    t("loading processor…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    t("loading model")
    # Grounding DINO has mixed components (Swin backbone + BERT text encoder + fusion).
    # BERT outputs float32, so loading in float16 causes dtype mismatches in fusion layers.
    # Load in float32 and move to GPU manually.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor

def make_batch(processor, bs=1):
    """
    Create a dummy image batch for Grounding DINO.
    Uses a random 640x480 RGB image repeated for the batch.
    """
    # Create a dummy RGB image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    images = [dummy_image] * bs
    text_prompts = [TEXT_PROMPT] * bs

    inputs = processor(images=images, text=text_prompts, return_tensors="pt")
    # Move all tensors to GPU (keep float32 to match model dtype)
    batch = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to("cuda")
        else:
            batch[k] = v
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
    # For detection models, report images/second as throughput
    n_images = int(batch["pixel_values"].shape[0])
    imgs_per_s = (n_images / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Images in batch: {n_images}")
    t(f"🚀 Throughput: {imgs_per_s:.2f} images/s")

    if return_metrics:
        return {"runtime_s": runtime_s, "n_images": n_images, "throughput_imgs_s": imgs_per_s}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
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
    model, processor = load_model(local_only=True)
    batch = make_batch(processor, bs=BATCH_SIZE)

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
    safe_model_id = MODEL_ID.replace("/", "_")
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
        throughputs.append(m["throughput_imgs_s"])
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
    print(f"Avg throughput:      {avg_tps:.2f} images/s")
    print(f"P50 throughput:      {p50_tps:.2f} images/s")
    print(f"P95 throughput:      {p95_tps:.2f} images/s")
    print(f"Std dev (imgs/s):    {stdev_tps:.2f}")
    print(f"Avg runtime:         {avg_runtime:.3f} s")
    print(inspect.getfile(model.__class__))


    t("done")

if __name__ == "__main__":
    main()
