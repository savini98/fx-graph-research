#!/usr/bin/env python3
"""Move cold-start / warm / launches / kernels to a batch=1 profiling pass.
Reverts the earlier big-batch instrumentation and inserts one call to
profiling_utils.profile_small_batch before the big-batch compile.
chronos is handled separately (custom pipeline)."""
import glob, os

REPO = os.path.dirname(os.path.abspath(__file__))

# what instrument_scripts.py inserted (to be reverted)
COLD_NEW = (
    "    import time as _time\n"
    "    torch.cuda.synchronize()\n"
    "    _cold_t0 = _time.perf_counter()\n"
    "    warmup(compiled, batch, iters=1)\n"
    "    torch.cuda.synchronize()\n"
    "    print(f\"Cold start (compile): {(_time.perf_counter() - _cold_t0) * 1000:.1f} ms\")\n"
)
COLD_OLD = "    warmup(compiled, batch, iters=1)\n"

METRICS_NEW = (
    "    hit, names = detect_cudagraphs(compiled, batch, trace=trace_path)\n"
    "    try:\n"
    "        import json as _json\n"
    "        with open(trace_path) as _tf:\n"
    "            _tr = _json.load(_tf)\n"
    "        _evs = _tr.get(\"traceEvents\", _tr if isinstance(_tr, list) else [])\n"
    "        _launch = sum(1 for _e in _evs if isinstance(_e, dict) and \"cudaGraphLaunch\" in _e.get(\"name\", \"\"))\n"
    "        _kern = sum(1 for _e in _evs if isinstance(_e, dict) and _e.get(\"cat\", \"\") == \"kernel\")\n"
    "        print(f\"CUDA graph launches: {_launch}\")\n"
    "        print(f\"Kernel count: {_kern}\")\n"
    "    except Exception as _ex:\n"
    "        print(f\"trace metric counting failed: {_ex}\")\n"
)
METRICS_OLD = "    hit, names = detect_cudagraphs(compiled, batch, trace=trace_path)\n"

COMPILE_ANCHOR = "    compiled = compile_model(model)\n"
PROFILE_CALL = (
    "    import sys as _sys\n"
    "    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), \"..\"))\n"
    "    from profiling_utils import profile_small_batch\n"
    "    profile_small_batch(model, batch, compile_model,\n"
    "                        os.path.join(os.path.dirname(os.path.abspath(__file__)), \"traces\", f\"profile_{TYPE}.json\"))\n"
    "    compiled = compile_model(model)\n"
)

for s in sorted(glob.glob(os.path.join(REPO, "models", "*", "*_script.py"))):
    mdir = os.path.basename(os.path.dirname(s))
    if mdir == "chronos-bolt-small":
        continue
    with open(s) as f:
        src = f.read()
    orig = src
    notes = []
    if COLD_NEW in src:
        src = src.replace(COLD_NEW, COLD_OLD, 1); notes.append("cold:reverted")
    if METRICS_NEW in src:
        src = src.replace(METRICS_NEW, METRICS_OLD, 1); notes.append("metrics:reverted")
    if "profile_small_batch" in src:
        notes.append("profile:already")
    elif COMPILE_ANCHOR in src:
        src = src.replace(COMPILE_ANCHOR, PROFILE_CALL, 1); notes.append("profile:inserted")
    else:
        notes.append("profile:NO-ANCHOR")
    if src != orig:
        with open(s, "w") as f:
            f.write(src)
    print(f"{mdir:32s} {' '.join(notes)}")
