# Print the name of the current conda environment
import os
import inspect
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "Not running in a conda environment"))

import os, time, argparse, torch, statistics as stats
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.profiler import profile, ProfilerActivity
import argparse




TYPE = "original"  # Set your desired type label here
BATCH_SIZE = 150  # Set your desired batch size here
MODEL_ID = "blenderbot-400M-distill"
OUTPUT_TOKEN_LENGTH = 100  # Set your desired output token length limit here
PROMPT = "Write a poem about AI"

# 1) Keep everything simple & fast
os.environ.pop("TORCH_LOGS", None)  # add "inductor,cudagraphs" later if you need diagnostics
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def t(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model(local_only=True):
    t("loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
    legacy=False)

    # # Ensure pad_token exists for decoder-only generation (avoid warnings / shape issues)
    # if tok.pad_token is None:
    #     tok.pad_token = tok.eos_token

    t("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Disable KV cache to keep allocations/static shapes simple
    # if hasattr(model.config, "use_cache"):
    #     model.config.use_cache = False
    return model, tok

def fixed_batch(tok, bs=1, seq_len=5):
    prompt = "Write a poem about AI"
    # Ensure pad_token exists for decoder-only generation (avoid warnings / shape issues)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
    # Repeat the prompt to match the batch size
    input_ids = input_ids.repeat(bs, 1)
    attention_mask = attention_mask.repeat(bs, 1)
    # For Seq2Seq models, need decoder_input_ids for forward pass
    # Use BOS token as start for decoder_input_ids
    if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
        bos_token_id = tok.bos_token_id
    else:
        bos_token_id = tok.eos_token_id
    decoder_input_ids = torch.full((input_ids.shape[0], 1), bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids
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
    # fn(**inp)
    # torch.cuda.synchronize()
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

def print_graph_breaks(model, inp):
    """
    Print only the number of graph breaks detected by torch._dynamo.explain.
    """
    t("checking Dynamo graph breaks…")
    try:
        import torch._dynamo as dynamo
        try:
            result = dynamo.explain(model)(**inp)
        except TypeError:
            result = dynamo.explain(model, inp)

        print(f"Graph breaks: {result}")
    except Exception as e:
        t(f"graph-break analysis failed: {e}.")

# ----------------------------- generation -----------------------------
@torch.inference_mode()
def generate_text(
    model,
    tok,
    batch,
    max_new_tokens=256,
    do_sample=True,
    eos_token_id=None,
    return_metrics=False,
):
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=eos_token_id if eos_token_id is not None else tok.eos_token_id,
    )

    # Count input tokens (prompt length)
    input_len = int(batch["input_ids"].shape[1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    out = model.generate(**batch, **gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    total_len = int(out.shape[1])
    # Only count raw generated tokens (exclude input tokens)
    new_tokens = max(0, total_len - input_len)
    runtime_s = end - start
    toks_per_s = (new_tokens / runtime_s) if runtime_s > 0 else float("inf")

    t(f"⏱ Runtime: {runtime_s:.3f} s")
    t(f"📝 Raw new tokens generated: {new_tokens}")
    t(f"🚀 Throughput: {toks_per_s:.2f} tokens/s")

    text = tok.decode(out[0], skip_special_tokens=False)
    if return_metrics:
        return text, {"runtime_s": runtime_s, "new_tokens": new_tokens, "throughput_tps": toks_per_s}
    return text

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='original', help='Type label for trace file (default: original)')
    parser.add_argument('--runs', type=int, default=30, help='Number of timed runs to perform (default: 30)')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size for generation (default: 150)')
    args = parser.parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    global TYPE
    TYPE = args.type
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    t("start")
    model, tok = load_model(local_only=True)
    batch = fixed_batch(tok, bs=BATCH_SIZE, seq_len=5)

    # Quick eager sanity check (no compile) — catches download/shape issues immediately
    t("eager forward sanity check…")
    # Remove decoder_inputs_embeds if present to avoid ValueError
    if "decoder_inputs_embeds" in batch:
        del batch["decoder_inputs_embeds"]
    with torch.inference_mode():
        _ = model(**batch)
        torch.cuda.synchronize()
    t("eager ok")

    # NEW: inspect graph breaks on the eager model before compiling
    # print_graph_breaks(model, batch)

    compiled = compile_model(model)
    warmup(compiled, batch, iters=1)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = f"traces/{MODEL_ID}_trace_{TYPE}_{dt_str}.json"
    hit, names = detect_cudagraphs(compiled, batch, trace=trace_path)
    if hit:
        t("✅ CUDA Graph activity detected:")
        for n in names: print("  -", n)
    else:
        t("❌ No CUDA Graph API events detected in this step.")

    # ------------------ benchmark runs ------------------
    import csv
    t(f"benchmarking {args.runs} timed runs…")
    throughputs = []
    runtimes = []
    new_tokens_list = []
    csv_rows = []

    for i in range(args.runs):
        t(f"run {i+1}/{args.runs}…")
        text, m = generate_text(
            compiled, tok, batch,
            max_new_tokens=OUTPUT_TOKEN_LENGTH,
            do_sample=False,
            return_metrics=True
        )
        throughputs.append(m["throughput_tps"])
        runtimes.append(m["runtime_s"])
        new_tokens_list.append(m["new_tokens"])
        csv_rows.append({
            "run": i+1,
            "throughput_tps": m["throughput_tps"],
            "runtime_s": m["runtime_s"],
            "new_tokens": m["new_tokens"]
        })

    # Save to CSV
    csv_filename = f"traces/{MODEL_ID}_benchmark_runs_{TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["run", "throughput_tps", "runtime_s", "new_tokens"])
        writer.writeheader()
        writer.writerows(csv_rows)
    t(f"Saved per-run data to {csv_filename}")

    # ------------------ summary stats ------------------
    avg_tps = sum(throughputs) / len(throughputs)
    p50_tps = stats.median(throughputs)
    stdev_tps = stats.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    p95_tps = sorted(throughputs)[max(0, int(round(0.95 * len(throughputs))) - 1)]
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_new_tokens = sum(new_tokens_list) / len(new_tokens_list)

    print("\n===== Benchmark summary =====")
    print(f"Timed runs:       {args.runs}")
    print(f"Avg throughput:   {avg_tps:.2f} tokens/s")
    print(f"P50 throughput:   {p50_tps:.2f} tokens/s")
    print(f"P95 throughput:   {p95_tps:.2f} tokens/s")
    print(f"Std dev (tps):    {stdev_tps:.2f}")
    print(f"Avg runtime:      {avg_runtime:.3f} s")
    print(f"Avg new tokens:   {avg_new_tokens:.1f}")
    print(inspect.getfile(compiled.__class__))
    print_graph_breaks(compiled, batch)
    

    t("done")

if __name__ == "__main__":
    main()
