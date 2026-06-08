"""Cold-start vs warm profiling, measured in-code (reliable, no trace segmentation).

Timed at the SAME (benchmark) batch as throughput. We compile WITHOUT a prior
warmup so the first timed forward is the genuine cold run (it pays the compile +
CUDA-graph capture). Subsequent runs are warm (steady-state replay).

Reports:
  Cold start (run1): <ms>          # first forward incl. compile + capture
  Warm forward (mean): <ms>        # mean of the remaining runs (steady state)
  CUDA graph launches: <n>         # in one warm forward
  Kernel count: <n>                # in one warm forward
"""
import os, time, json, statistics
import torch
from torch.profiler import profile, ProfilerActivity


def profile_small_batch(model, batch, compile_fn, trace_path, runs=30):
    """`batch` is the validated (post eager-check) forward batch at the
    benchmark size. Measures cold (run 1) and warm (mean of the rest)."""
    import torch._dynamo as _dyn
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    _dyn.reset()
    try:
        fn = compile_fn(model)
        times = []
        with torch.inference_mode():
            for _ in range(runs):
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
                fn(**batch)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - _t0) * 1000.0)

        cold = times[0]
        # Median of the post-cold runs = steady state, robust to the few slow
        # CUDA-graph *capture* runs that follow the cold compile.
        warm_runs = times[1:]
        warm = statistics.median(warm_runs) if warm_runs else float("nan")
        print(f"Cold start (run1): {cold:.1f} ms", flush=True)
        print(f"Warm forward (median): {warm:.4f} ms", flush=True)
        print(f"Warm forward (mean): {statistics.mean(warm_runs):.4f} ms", flush=True)

        # launches + kernels from ONE warm (post-cold) profiled forward
        try:
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                         with_stack=False) as prof:
                with torch.inference_mode():
                    fn(**batch)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                tr = json.load(f)
            evs = tr.get("traceEvents", tr if isinstance(tr, list) else [])
            launch = sum(1 for e in evs if isinstance(e, dict) and "cudaGraphLaunch" in e.get("name", ""))
            kern = sum(1 for e in evs if isinstance(e, dict) and e.get("cat", "") == "kernel")
            syncs = sum(1 for e in evs if isinstance(e, dict) and "Synchronize" in e.get("name", ""))
            print(f"CUDA graph launches: {launch}", flush=True)
            print(f"Kernel count: {kern}", flush=True)
            print(f"Sync count: {syncs}", flush=True)
        except Exception as ex:
            print(f"profile metric counting failed: {ex}", flush=True)
    except Exception as ex:
        print(f"profiling failed: {ex}", flush=True)
    finally:
        _dyn.reset()
