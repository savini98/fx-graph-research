#!/usr/bin/env python3
"""Fix graph-break reporting in biogpt / phi-4-mini / qwen-audio-chat:
- phi & qwen have an active but broken print_graph_breaks (wrong explain API);
  replace its body with the correct dynamo.explain(model)(**inp) form.
- biogpt's print_graph_breaks is fully commented; add a correct one.
- all three: uncomment the `# print_graph_breaks(model, batch)` call.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))

BROKEN = (
    "        try:\n"
    "            result = dynamo.explain(model, **inp)\n"
    "        except TypeError:\n"
    "            result = dynamo.explain(model, inp)\n"
    "\n"
    "        breaks = getattr(result, \"graph_breaks\", [])\n"
    "        print(f\"Graph breaks: {len(breaks)}\")\n"
)
FIXED = (
    "        explanation = dynamo.explain(model)(**inp)\n"
    "        print(\"\\n===== Graph Break Report =====\")\n"
    "        print(f\"Graph break count: {explanation.graph_break_count}\")\n"
    "        print(f\"Number of graphs:  {explanation.graph_count}\")\n"
    "        print(\"=\" * 35)\n"
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

CALL_OLD = "    # print_graph_breaks(model, batch)\n"
CALL_NEW = "    print_graph_breaks(model, batch)\n"

targets = {
    "phi-4-mini": "phi_4_mini_script.py",
    "qwen-audio-chat": "qwen_audio_chat_script.py",
    "biogpt": "biogpt_script.py",
}

for mdir, fn in targets.items():
    p = os.path.join(REPO, "models", mdir, fn)
    with open(p) as f:
        src = f.read()
    notes = []

    # phi/qwen: fix the broken explain body
    if BROKEN in src:
        src = src.replace(BROKEN, FIXED, 1); notes.append("body:fixed")

    # biogpt: no active function -> add one (its existing def is commented out)
    has_active_def = any(
        ln.lstrip().startswith("def print_graph_breaks") for ln in src.splitlines()
    )
    if not has_active_def:
        src = src.replace("def main():", GB_FUNC + "def main():", 1); notes.append("func:added")

    # all three: uncomment the call
    if CALL_OLD in src:
        src = src.replace(CALL_OLD, CALL_NEW, 1); notes.append("call:uncommented")
    elif CALL_NEW in src:
        notes.append("call:already")
    else:
        notes.append("call:NO-ANCHOR")

    with open(p, "w") as f:
        f.write(src)
    print(f"{mdir:20s} {' '.join(notes)}")
