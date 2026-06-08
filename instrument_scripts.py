#!/usr/bin/env python3
"""One-off: instrument the benchmark scripts to capture cold-start (compile)
time, CUDA-graph launches and kernel counts, and (for the 6 that lack it) the
graph-break report. Idempotent — skips a script that already has the markers.
"""
import glob, os

REPO = os.path.dirname(os.path.abspath(__file__))

COLD_OLD = "    warmup(compiled, batch, iters=1)\n"
COLD_NEW = (
    "    import time as _time\n"
    "    torch.cuda.synchronize()\n"
    "    _cold_t0 = _time.perf_counter()\n"
    "    warmup(compiled, batch, iters=1)\n"
    "    torch.cuda.synchronize()\n"
    "    print(f\"Cold start (compile): {(_time.perf_counter() - _cold_t0) * 1000:.1f} ms\")\n"
)

METRICS_ANCHOR = "    hit, names = detect_cudagraphs(compiled, batch, trace=trace_path)\n"
METRICS_NEW = METRICS_ANCHOR + (
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

GB_FUNC = (
    "\n"
    "def print_graph_breaks(model, inp):\n"
    "    t(\"checking Dynamo graph breaks\\u2026\")\n"
    "    try:\n"
    "        import torch._dynamo as dynamo\n"
    "        explanation = dynamo.explain(model)(**inp)\n"
    "        print(\"\\n===== Graph Break Report =====\")\n"
    "        print(f\"Graph break count: {explanation.graph_break_count}\")\n"
    "        print(f\"Number of graphs:  {explanation.graph_count}\")\n"
    "        print(\"=\" * 35)\n"
    "    except Exception as e:\n"
    "        t(f\"graph-break analysis failed: {e}\")\n"
    "\n"
)
GB_CALL_OLD = "    compiled = compile_model(model)\n"
GB_CALL_NEW = "    print_graph_breaks(model, batch)\n    compiled = compile_model(model)\n"

SIX = {"biogpt", "blenderbot-400M-distill", "flan-t5-large",
       "longformer-scico", "phi-4-mini", "qwen-audio-chat"}

patched = []
for s in sorted(glob.glob(os.path.join(REPO, "models", "*", "*_script.py"))):
    mdir = os.path.basename(os.path.dirname(s))
    with open(s) as f:
        src = f.read()
    orig = src
    notes = []

    if "Cold start (compile)" in src:
        notes.append("cold:already")
    elif COLD_OLD in src:
        src = src.replace(COLD_OLD, COLD_NEW, 1); notes.append("cold:ok")
    else:
        notes.append("cold:NO-ANCHOR")

    if "CUDA graph launches" in src:
        notes.append("metrics:already")
    elif METRICS_ANCHOR in src:
        src = src.replace(METRICS_ANCHOR, METRICS_NEW, 1); notes.append("metrics:ok")
    else:
        notes.append("metrics:NO-ANCHOR")

    if mdir in SIX:
        if "def print_graph_breaks" in src:
            notes.append("gb:already")
        elif "def main():" in src and GB_CALL_OLD in src:
            src = src.replace("def main():", GB_FUNC + "def main():", 1)
            src = src.replace(GB_CALL_OLD, GB_CALL_NEW, 1)
            notes.append("gb:ok")
        else:
            notes.append("gb:NO-ANCHOR")

    if src != orig:
        with open(s, "w") as f:
            f.write(src)
        patched.append(mdir)
    print(f"{mdir:32s} {' '.join(notes)}")

print(f"\nPatched {len(patched)} scripts.")
