"""Cold-start + warm + launches/kernels, traced FROM THE VERY FIRST RUN.

Matches the paper methodology (verified: region-0/0 run-1 reproduces the paper's
3090 bart-base 2238->189 ms = 12x exactly). We compile and profile with NO prior
warmup, so the first "Torch-Compiled Region: 0/0" run is the genuine cold run
(compile + CUDA-graph capture). Subsequent 0/0 runs are warm (replay).

IMPORTANT: call this BEFORE anything else compiles the model (e.g. before
print_graph_breaks / dynamo.explain), otherwise run-1 is no longer cold.

Prints: Cold start (run1) ms, Warm forward (median) ms, CUDA graph launches,
Kernel count, Sync count.
"""
import os, json, statistics
import torch
from torch.profiler import profile, ProfilerActivity


def _region_runs(evs):
    reg = sorted(
        [e for e in evs if isinstance(e, dict) and "ts" in e
         and str(e.get("name", "")).strip() == "Torch-Compiled Region: 0/0"],
        key=lambda e: e["ts"])
    return [(reg[i + 1]["ts"] - reg[i]["ts"]) / 1000.0 for i in range(len(reg) - 1)]


def profile_small_batch(model, batch, compile_fn, trace_path, runs=8):
    import torch._dynamo as _dyn
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    _dyn.reset()
    try:
        fn = compile_fn(model)
        # Trace from the FIRST run — no warmup. run1 = compile + capture (cold).
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                     with_stack=False) as prof:
            with torch.inference_mode():
                for _ in range(runs):
                    fn(**batch)
                    torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            evs = json.load(f).get("traceEvents", [])

        run_ms = _region_runs(evs)
        if run_ms:
            cold = run_ms[0]
            warm = statistics.median(run_ms[1:]) if len(run_ms) > 1 else float("nan")
            print(f"Cold start (run1): {cold:.1f} ms", flush=True)
            print(f"Warm forward (median): {warm:.4f} ms", flush=True)
        else:
            print("Cold start (run1): NA (no 0/0 regions)", flush=True)

        # per-forward launch/kernel/sync counts from the LAST (warm) region window
        reg = sorted([e for e in evs if isinstance(e, dict) and "ts" in e
                      and str(e.get("name", "")).strip() == "Torch-Compiled Region: 0/0"],
                     key=lambda e: e["ts"])
        lo = reg[-1]["ts"] if len(reg) >= 1 else 0
        win = [e for e in evs if isinstance(e, dict) and e.get("ts", 0) >= lo]
        launch = sum(1 for e in win if "cudaGraphLaunch" in e.get("name", ""))
        kern = sum(1 for e in win if e.get("cat", "") == "kernel")
        syncs = sum(1 for e in win if "Synchronize" in e.get("name", ""))
        print(f"CUDA graph launches: {launch}", flush=True)
        print(f"Kernel count: {kern}", flush=True)
        print(f"Sync count: {syncs}", flush=True)
    except Exception as ex:
        print(f"profiling failed: {ex}", flush=True)
    finally:
        _dyn.reset()
