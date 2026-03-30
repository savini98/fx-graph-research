"""
Shared GPU utilities for all model benchmark scripts.
Provides auto batch size detection to maximize GPU utilization without OOM.
"""
import torch
import gc


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

    # Available memory after reserving for:
    # - Model weights (baseline_mem)
    # - torch.compile inductor overhead
    # - CUDA graph private memory pools
    # - KV cache growth during generation (grows with each token step)
    # - dynamo.explain tracing overhead
    available_mem = (total_mem * target_utilization) - baseline_mem
    if available_mem <= 0:
        # Model itself takes most of the GPU — use minimum batch size
        if verbose:
            print(f"[GPU] Model uses {baseline_mem / (1024**3):.1f} GB, "
                  f"barely fits. Using batch_size={min_bs}", flush=True)
        return min_bs

    estimated_bs = int(available_mem / per_sample_mem)
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
