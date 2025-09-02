import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

# ==== Config ====
# Set to "microsoft/Phi-4-mini-instruct" or to a local snapshot directory containing a valid config.json
MODEL_ID = "Phi-4-mini-instruct"
NUM_RUNS = 15                      # total runs; the "middle" one will be profiled
MAX_NEW_TOKENS = 50              # cap generated length
USE_NONCE_IN_PROMPT = False       # set True to append a tiny nonce to prompt
TEMPERATURE_SCHEDULE = [0.8, 0.9, 1.0, 0.7, 1.1]
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.08

# (Optional) allow TF32 on Ampere or newer for extra GEMM speed
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ---- Load ----
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
)

# Ensure pad_token exists for decoder-only generation (avoid warnings / shape issues)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

compiled_model = torch.compile(model)

# ---- Prompt ----
BASE_PROMPT = "I went to school today and..... Complete this into a short story."

# ==== Helpers ====
def count_new_tokens(generate_out, inputs, model):
    """
    Return the number of *new* tokens produced by .generate().
    Decoder-only: sequences include the prompt; subtract true prompt length from attention_mask.
    """
    sequences = generate_out.sequences  # [batch, seq_len]
    eos_id = model.config.eos_token_id
    eos_ids = {eos_id} if eos_id is not None and not isinstance(eos_id, (list, tuple)) else set(eos_id or [])

    # Decoder-only branch
    attn = inputs.get("attention_mask", None)
    if attn is not None:
        prompt_lens = attn.sum(dim=1)
    else:
        prompt_lens = torch.tensor([inputs["input_ids"].shape[1]] * sequences.shape[0], device=sequences.device)
    per_item = sequences.shape[1] - prompt_lens

    # Optionally exclude trailing EOS if present
    end_is_eos = torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)
    if eos_ids:
        for eid in eos_ids:
            end_is_eos |= (sequences[:, -1] == eid)
    per_item = per_item - end_is_eos.int()
    return int(per_item.sum().item())

def run_one_generation(model, prompt_text, max_new_tokens, temperature, seed):
    """Generate once with sampling; return (gen_out, elapsed_seconds, new_tokens, decoded_text)."""
    device = next(model.parameters()).device

    # Tighter tokenization: no huge padding to model_max_length
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    # Manually pad to batch if needed (batch=1 here, so fine), then move to device:
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set per-run seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=temperature,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    new_tokens = count_new_tokens(gen_out, inputs, model)
    text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
    return gen_out, elapsed, new_tokens, text

def profile_run(model, run_idx_to_profile, prompt_text, temperature):
    """Profile exactly one run (CPU+CUDA) — the middle run — and write a trace."""
    device = next(model.parameters()).device
    os.makedirs('./profiler_logs', exist_ok=True)

    # Fresh seed for the profiled run
    seed = int(time.time() * 1e6) % (2**31 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare inputs outside the profiler to avoid capturing tokenization overhead
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    print(f"\n[Profiler] Capturing run {run_idx_to_profile+1} only (CPU+CUDA). Seed={seed}, Temp={temperature}")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Profile exactly the generation call
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=TOP_P,
                top_k=TOP_K,
                temperature=temperature,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        prof.step()  # optional boundary mark
        # prof.export_chrome_trace("trace.json")

    # Export a single-run chrome trace (openable in chrome://tracing)
    trace_path = f'./profiler_logs/run_{MODEL_ID}_{run_idx_to_profile+1}.json'
    try:
        prof.export_chrome_trace(trace_path)
        print(f"[Profiler] Exported single-run trace to: {trace_path}")
    except Exception:
        print("[Profiler] Could not export chrome trace (safe to ignore).")

    # Print top kernels/operators for this one run
    print(f"\n--- compiled_model CUDA Time (middle run only) ---")
    try:
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    except Exception:
        print("No CUDA profile available (CPU-only run).")

    print(f"\n--- compiled_model Memory Usage (middle run only) ---")
    print(prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=5))

    # Return details to keep reporting consistent
    new_tokens = count_new_tokens(gen_out, inputs, model)
    text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
    return elapsed, new_tokens, text

def print_graph_breaks(model):
    device = next(model.parameters()).device
    enc = tokenizer(
        BASE_PROMPT,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    print("\n=== Graph Breaks Analysis (TorchDynamo) ===")
    try:
        # Decoder-only: no decoder_input_ids needed
        explanation = torch._dynamo.explain(model)(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None)
        )
        print(explanation)
    except Exception as e:
        print(f"Graph break analysis failed: {e}")

# ==== Warmup (outside any profiler) ====
device = next(compiled_model.parameters()).device
warm_inputs = tokenizer(
    BASE_PROMPT,
    return_tensors="pt",
    padding=False,
    truncation=True,
)
warm_inputs = {k: v.to(device) for k, v in warm_inputs.items()}
with torch.no_grad():
    _ = compiled_model.generate(
        **warm_inputs,
        max_new_tokens=min(50, MAX_NEW_TOKENS),
        do_sample=False,  # warm caches deterministically
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

# ==== Regular runs for timing stats (skip first run, do NUM_RUNS + 1 total) ====
total_time = 0.0
total_new_tokens = 0

print("\n=== Running regular timing runs (first run skipped for warmup) ===")
for run_idx in range(NUM_RUNS + 1):  # Add an extra run since we're skipping the first
    # Optionally vary the prompt with a nonce
    prompt = BASE_PROMPT
    if USE_NONCE_IN_PROMPT:
        nonce = torch.randint(0, 1_000_000, ()).item()
        prompt = f"{BASE_PROMPT} [session:{nonce}]"

    temperature = TEMPERATURE_SCHEDULE[run_idx % len(TEMPERATURE_SCHEDULE)]

    # Regular run
    seed = int(time.time() * 1e6) % (2**31 - 1)
    if run_idx == NUM_RUNS:
      # [NSYS] last run, start nsys
      torch.cuda.cudart().cudaProfilerStart()
    _, elapsed, new_tokens, text = run_one_generation(
        compiled_model, prompt, MAX_NEW_TOKENS, temperature, seed
    )
    if run_idx == NUM_RUNS:
      # [NSYS] last run, stop nsys
      torch.cuda.cudart().cudaProfilerStop()

    # Skip recording stats for the first run (run_idx == 0)
    if run_idx > 0:
        total_time += elapsed
        total_new_tokens += new_tokens
    tps = (new_tokens / elapsed) if elapsed > 0 else float('inf')

    print(f"\n--- Run {run_idx+1} / {NUM_RUNS} ---")
    print(f"Temp: {temperature}")
    print(f"Generated {new_tokens} new tokens in {elapsed:.3f}s ({tps:.2f} tok/s)")
    print("Output:")
    print(text)

# ==== Aggregate stats ====
avg_time = total_time / NUM_RUNS
time_per_token = (total_time / total_new_tokens) if total_new_tokens > 0 else float("inf")
toks_per_sec = (total_new_tokens / total_time) if total_time > 0 else 0.0
avg_toks_per_run = total_new_tokens / NUM_RUNS

# # ==== Separate profiler run ====
# print("\n=== Running separate profiler run ===")
# prompt = BASE_PROMPT
# if USE_NONCE_IN_PROMPT:
#     nonce = torch.randint(0, 1_000_000, ()).item()
#     prompt = f"{BASE_PROMPT} [session:{nonce}]"

# temperature = TEMPERATURE_SCHEDULE[0]  # Use first temperature for profiler run
# elapsed, new_tokens, text = profile_run(compiled_model, 0, prompt, temperature)

# print(f"\n--- Profiler Run Results ---")
# print(f"Temp: {temperature}")
# print(f"Generated {new_tokens} new tokens in {elapsed:.3f}s ({(new_tokens/elapsed):.2f} tok/s)")
# print("Output:")
# print(text)

print(f"\n=== Aggregate Stats (all runs; only middle run profiled) ===")
print(f"Average runtime over {NUM_RUNS} runs: {avg_time:.4f} seconds")
print(f"Total generated tokens: {total_new_tokens}")
print(f"Average generated tokens/run: {avg_toks_per_run:.2f}")
print(f"Average runtime per token: {time_per_token:.6f} s/token  ({toks_per_sec:.2f} tok/s)")

# ==== Graph breaks ====
# print_graph_breaks(compiled_model)

# # Optional GPU mem snapshot
# if torch.cuda.is_available():
#     print("\n=== GPU Memory Usage Snapshot ===")
#     print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#     print(f"Reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
#     print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# print("\nIf you need a visual trace, open the Chrome trace file (single profiled run):")
# print("  chrome://tracing  ->  Load ./profiler_logs/middle_run_*.json")
