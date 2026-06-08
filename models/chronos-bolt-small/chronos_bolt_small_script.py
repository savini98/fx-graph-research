# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats
from datetime import datetime
from torch.profiler import profile, ProfilerActivity
import argparse
import numpy as np




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 64  # Set your desired batch size here
MODEL_ID = "autogluon/chronos-bolt-small"
PREDICTION_LENGTH = 24  # Forecast 24 steps ahead

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(SCRIPT_DIR, "traces")

# 1) Keep everything simple & fast
os.environ.pop("TORCH_LOGS", None)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model():
    from chronos import BaseChronosPipeline

    t("loading chronos-bolt-small pipeline…")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    return pipeline

def make_batch(bs=1, seq_len=128):
    """
    Create a batch of dummy time series data for Chronos-Bolt.
    Shape: (batch_size, seq_len) — synthetic random walk data.
    """
    # Generate synthetic random walk time series
    np.random.seed(42)
    data = np.cumsum(np.random.randn(bs, seq_len), axis=1).astype(np.float32)
    context = torch.tensor(data)
    return context

def get_inner_model(pipeline):
    """
    Extract the underlying T5 model from the Chronos pipeline for graph break analysis.
    """
    # Chronos wraps a T5 model internally
    if hasattr(pipeline, 'model'):
        return pipeline.model
    if hasattr(pipeline, 'inner_model'):
        return pipeline.inner_model
    return pipeline

def compile_model(m):
    t("compiling with torch.compile (inductor, reduce-overhead)…")
    return torch.compile(m, backend="inductor", mode="reduce-overhead", fullgraph=False)

@torch.inference_mode()
def warmup_pipeline(pipeline, context, prediction_length, iters=1):
    t(f"warmup x{iters}…")
    for _ in range(iters):
        pipeline.predict(inputs=context, prediction_length=prediction_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

@torch.inference_mode()
def detect_cudagraphs_pipeline(pipeline, context, prediction_length, trace="trace.json"):
    t("profiling one step…")
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=False) as prof:
        for _ in range(7):
            pipeline.predict(inputs=context, prediction_length=prediction_length)
            if torch.cuda.is_available():
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
def print_graph_breaks(pipeline, context, prediction_length):
    """
    Use torch._dynamo.explain to report graph break count and reasons.
    Chronos uses a pipeline.predict() API, so we need to trace the inner model.
    """
    t("checking Dynamo graph breaks…")
    try:
        import torch._dynamo as dynamo

        inner_model = get_inner_model(pipeline)
        t(f"inner model class: {inner_model.__class__.__name__}")

        # Try to get a forward-pass compatible input from the pipeline
        # First, do a normal predict to see if we can intercept the inner model call
        # Use dynamo.explain on the inner model with tokenized inputs
        if hasattr(pipeline, 'embed_and_tokenize'):
            # Chronos-Bolt specific: get tokenized input
            token_ids, attention_mask, scale = pipeline.embed_and_tokenize(context.to(
                pipeline.device if hasattr(pipeline, 'device') else 'cuda'
            ))
            inp = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
            }
            # Add decoder_input_ids for T5
            if hasattr(inner_model.config, "decoder_start_token_id"):
                bos_id = inner_model.config.decoder_start_token_id or 0
            else:
                bos_id = 0
            inp["decoder_input_ids"] = torch.full(
                (token_ids.shape[0], 1), bos_id,
                dtype=token_ids.dtype, device=token_ids.device
            )
            explanation = dynamo.explain(inner_model)(**inp)
        else:
            # Fallback: explain the whole pipeline predict
            explanation = dynamo.explain(lambda ctx: pipeline.predict(inputs=ctx, prediction_length=prediction_length))(context)

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
        import traceback
        traceback.print_exc()

# ----------------------------- benchmark -----------------------------
@torch.inference_mode()
def run_predict(
    pipeline,
    context,
    prediction_length,
    return_metrics=False,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    forecast = pipeline.predict(inputs=context, prediction_length=prediction_length)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    runtime_s = end - start
    n_series = int(context.shape[0])
    series_per_s = (n_series / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Series in batch: {n_series}")
    t(f"🚀 Throughput: {series_per_s:.2f} series/s")

    if return_metrics:
        return {"runtime_s": runtime_s, "n_series": n_series, "throughput_series_s": series_per_s}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64, 0 not supported for chronos)')
    args = parser.parse_args()
    global BATCH_SIZE
    # Chronos uses a custom pipeline, not standard model(**batch),
    # so auto-detect (batch_size=0) is not supported. Use default.
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else 64
    global TYPE
    TYPE = args.type
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    t("start")
    os.makedirs(TRACES_DIR, exist_ok=True)
    pipeline = load_model()
    context = make_batch(bs=BATCH_SIZE, seq_len=128)

    # Quick eager sanity check
    t("eager predict sanity check…")
    with torch.inference_mode():
        forecast = pipeline.predict(inputs=context, prediction_length=PREDICTION_LENGTH)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t(f"eager ok — forecast shape: {forecast.shape}")

    # Detect and report graph breaks
    print_graph_breaks(pipeline, context, PREDICTION_LENGTH)

    # Compile the inner model with torch.compile for actual PyTorch 2 optimization
    t("compiling inner model with torch.compile (inductor, reduce-overhead)…")
    pipeline.model = torch.compile(pipeline.model, backend="inductor", mode="reduce-overhead", fullgraph=False)

    # Warmup (triggers compilation on first run)
    import time as _time
    torch.cuda.synchronize()
    _cold_t0 = _time.perf_counter()
    warmup_pipeline(pipeline, context, PREDICTION_LENGTH, iters=1)
    torch.cuda.synchronize()
    print(f"Cold start (compile): {(_time.perf_counter() - _cold_t0) * 1000:.1f} ms")

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id = os.path.basename(MODEL_ID)
    trace_path = os.path.join(TRACES_DIR, f"{safe_model_id}_trace_{TYPE}_{dt_str}.json")
    hit, names = detect_cudagraphs_pipeline(pipeline, context, PREDICTION_LENGTH, trace=trace_path)
    try:
        import json as _json
        with open(trace_path) as _tf:
            _tr = _json.load(_tf)
        _evs = _tr.get("traceEvents", _tr if isinstance(_tr, list) else [])
        _launch = sum(1 for _e in _evs if isinstance(_e, dict) and "cudaGraphLaunch" in _e.get("name", ""))
        _kern = sum(1 for _e in _evs if isinstance(_e, dict) and _e.get("cat", "") == "kernel")
        print(f"CUDA graph launches: {_launch}")
        print(f"Kernel count: {_kern}")
    except Exception as _ex:
        print(f"trace metric counting failed: {_ex}")
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
        m = run_predict(pipeline, context, PREDICTION_LENGTH, return_metrics=True)
        throughputs.append(m["throughput_series_s"])
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
    print(f"Prediction length:   {PREDICTION_LENGTH}")
    print(f"Avg throughput:      {avg_tps:.2f} series/s")
    print(f"P50 throughput:      {p50_tps:.2f} series/s")
    print(f"P95 throughput:      {p95_tps:.2f} series/s")
    print(f"Std dev (series/s):  {stdev_tps:.2f}")
    print(f"Avg runtime:         {avg_runtime:.3f} s")
    inner_model = get_inner_model(pipeline)
    print(inspect.getfile(inner_model.__class__))


    t("done")

if __name__ == "__main__":
    main()
