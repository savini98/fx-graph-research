"""
Shared GPU utilities for all model benchmark scripts.
Provides auto batch size detection to maximize GPU utilization without OOM.
"""
import torch
import gc


def find_max_batch_size(make_batch_fn, model_fn, min_bs=1, max_bs=2048,
                        target_utilization=0.85, verbose=True):
    """
    Find the largest batch size that uses ~target_utilization of GPU memory.

    Instead of just checking if a forward pass fits, this measures actual memory
    usage per sample and calculates the batch size that fills the GPU to the
    target utilization level. This accounts for torch.compile overhead, KV cache
    growth during generation, and CUDA graph memory.

    Args:
        make_batch_fn: callable(batch_size) -> dict of tensors (the batch)
        model_fn: callable(**batch) -> output (forward pass)
        min_bs: minimum batch size to try
        max_bs: upper bound for search
        target_utilization: fraction of GPU memory to target (default 0.85)
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
            # Even probe_bs is too big, try with 1
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
    per_sample_mem = forward_mem / probe_bs

    # Available memory = total * target_utilization - baseline
    # Account for torch.compile overhead (~20% extra on top of forward pass)
    compile_overhead_factor = 1.3  # torch.compile + CUDA graphs need ~30% more
    available_mem = (total_mem * target_utilization) - baseline_mem
    estimated_bs = int(available_mem / (per_sample_mem * compile_overhead_factor))
    estimated_bs = max(min_bs, min(estimated_bs, max_bs))

    if verbose:
        print(f"[GPU] Probe batch_size={probe_bs}: peak={peak_mem / (1024**3):.2f} GB, "
              f"per_sample={per_sample_mem / (1024**2):.1f} MB", flush=True)
        print(f"[GPU] Available for batches: {available_mem / (1024**3):.2f} GB "
              f"(target {target_utilization*100:.0f}% of {total_mem / (1024**3):.1f} GB)", flush=True)
        print(f"[GPU] Estimated max batch size: {estimated_bs} "
              f"(with {compile_overhead_factor:.0%} compile overhead factor)", flush=True)

    # Step 4: Verify the estimated batch size actually fits
    torch.cuda.empty_cache()
    gc.collect()
    try:
        batch = make_batch_fn(estimated_bs)
        with torch.inference_mode():
            _ = model_fn(**batch)
            torch.cuda.synchronize()
        del batch, _
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print(f"[GPU] Verified: batch_size={estimated_bs} fits ✅", flush=True)
        return estimated_bs
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            # Fall back: binary search downward from estimate
            torch.cuda.empty_cache()
            gc.collect()
            safe_bs = max(min_bs, estimated_bs // 2)
            if verbose:
                print(f"[GPU] batch_size={estimated_bs} OOM'd, falling back to {safe_bs}", flush=True)
            return safe_bs
        raise
