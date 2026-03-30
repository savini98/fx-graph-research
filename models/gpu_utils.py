"""
Shared GPU utilities for all model benchmark scripts.
Provides auto batch size detection and safe graph break analysis.
"""
import torch
import gc


def safe_explain(model, batch, make_small_batch_fn=None, verbose_fn=print):
    """
    Run torch._dynamo.explain with OOM protection.
    If the full batch OOMs during tracing, retries with batch_size=2.

    Args:
        model: the model to explain
        batch: the full batch dict
        make_small_batch_fn: optional callable(bs) -> small batch for retry
        verbose_fn: print function (e.g. the script's t() function)
    """
    import torch._dynamo as dynamo

    try:
        explanation = dynamo.explain(model)(**batch)
        return explanation
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            if make_small_batch_fn is not None:
                verbose_fn("dynamo.explain OOM'd, retrying with small batch...")
                small = make_small_batch_fn(2)
                explanation = dynamo.explain(model)(**small)
                del small
                torch.cuda.empty_cache()
                gc.collect()
                return explanation
            else:
                raise
        raise


def find_max_batch_size(make_batch_fn, model_fn, min_bs=1, max_bs=2048,
                        target_utilization=0.70, verbose=True):
    """
    Find the largest batch size that uses ~target_utilization of GPU memory.

    Measures actual memory usage per sample and calculates the batch size that
    fills the GPU to the target utilization level. Uses a conservative target
    because torch.compile + CUDA graphs + generation KV cache all add significant
    overhead beyond what a single forward pass measures.

    Args:
        make_batch_fn: callable(batch_size) -> dict of tensors (the batch)
        model_fn: callable(**batch) -> output (forward pass)
        min_bs: minimum batch size to try
        max_bs: upper bound for search
        target_utilization: fraction of GPU memory to target (default 0.70)
            - 0.70 accounts for torch.compile overhead (~30%), CUDA graph
              private pools, and KV cache growth during generation
        verbose: print progress

    Returns:
        int: the batch size that should maximize GPU utilization without OOM
    """
    total_mem = torch.cuda.get_device_properties(0).total_memory
    if verbose:
        print(f"[GPU] {torch.cuda.get_device_name(0)}, {total_mem / (1024**3):.1f} GB VRAM", flush=True)

    # Step 1: Measure baseline memory (model weights + framework overhead)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated(0)
    if verbose:
        print(f"[GPU] Baseline memory (model loaded): {baseline_mem / (1024**3):.2f} GB", flush=True)

    # Step 2: Run a small batch to measure per-sample memory cost
    probe_bs = min(8, max_bs)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        batch = make_batch_fn(probe_bs)
        with torch.inference_mode():
            _ = model_fn(**batch)
            torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(0)
        del batch, _
        torch.cuda.empty_cache()
        gc.collect()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            probe_bs = 1
            torch.cuda.reset_peak_memory_stats()
            batch = make_batch_fn(1)
            with torch.inference_mode():
                _ = model_fn(**batch)
                torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(0)
            del batch, _
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise

    # Step 3: Calculate per-sample memory and target batch size
    forward_mem = peak_mem - baseline_mem
    per_sample_mem = max(forward_mem / probe_bs, 1024)  # at least 1KB to avoid division issues

    # A single forward pass with 1 decoder token massively underestimates generation
    # memory. During generation, KV cache grows linearly with output tokens and each
    # decoder layer stores key/value tensors for ALL previous tokens.
    # Empirical multiplier: generation uses ~10-50x more per-sample memory than
    # a single forward pass, depending on model depth and output length.
    # For safety, multiply per-sample cost by model-size-dependent factor.
    model_gb = baseline_mem / (1024**3)
    if model_gb > 3.0:
        # Large models (t5-3b, whisper-large-v3): generation KV cache is huge
        gen_multiplier = 80.0
    elif model_gb > 1.0:
        # Medium-large models (Florence-2-large)
        gen_multiplier = 40.0
    elif model_gb > 0.3:
        # Medium models (bart-base, bart-large-cnn, rebel-large, whisper-small)
        # CUDA graphs + generation KV cache + torch.compile reservations
        gen_multiplier = 30.0
    else:
        # Small models (t5-small, opus-mt, encoder-only)
        gen_multiplier = 10.0

    effective_per_sample = per_sample_mem * gen_multiplier

    # Available memory after reserving for model weights
    available_mem = (total_mem * target_utilization) - baseline_mem
    if available_mem <= 0:
        if verbose:
            print(f"[GPU] Model uses {model_gb:.1f} GB, "
                  f"barely fits. Using batch_size={min_bs}", flush=True)
        return min_bs

    estimated_bs = int(available_mem / effective_per_sample)
    estimated_bs = max(min_bs, min(estimated_bs, max_bs))

    if verbose:
        print(f"[GPU] Probe batch_size={probe_bs}: peak={peak_mem / (1024**3):.2f} GB, "
              f"per_sample={per_sample_mem / (1024**2):.1f} MB", flush=True)
        print(f"[GPU] Available for batches: {available_mem / (1024**3):.2f} GB "
              f"(target {target_utilization*100:.0f}% utilization)", flush=True)
        print(f"[GPU] Estimated batch size: {estimated_bs}", flush=True)

    # Step 4: Verify the estimated batch size actually fits
    torch.cuda.empty_cache()
    gc.collect()
    try:
        batch = make_batch_fn(estimated_bs)
        with torch.inference_mode():
            _ = model_fn(**batch)
            torch.cuda.synchronize()
        actual_peak = torch.cuda.max_memory_allocated(0)
        del batch, _
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print(f"[GPU] Verified: batch_size={estimated_bs} fits "
                  f"(peak {actual_peak / (1024**3):.2f} GB) ✅", flush=True)
        return estimated_bs
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            # Binary search downward
            torch.cuda.empty_cache()
            gc.collect()
            lo, hi = min_bs, estimated_bs - 1
            best = min_bs
            while lo <= hi:
                mid = (lo + hi) // 2
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    batch = make_batch_fn(mid)
                    with torch.inference_mode():
                        _ = model_fn(**batch)
                        torch.cuda.synchronize()
                    del batch, _
                    torch.cuda.empty_cache()
                    gc.collect()
                    best = mid
                    lo = mid + 1
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    gc.collect()
                    hi = mid - 1
            if verbose:
                print(f"[GPU] Estimate OOM'd, binary search found: batch_size={best}", flush=True)
            return best
        raise
