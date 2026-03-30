"""
Shared GPU utilities for all model benchmark scripts.
Provides auto batch size detection to maximize GPU utilization without OOM.
"""
import torch
import gc


def find_max_batch_size(make_batch_fn, model_fn, min_bs=1, max_bs=1024, verbose=True):
    """
    Binary search for the largest batch size that fits in GPU memory.

    Args:
        make_batch_fn: callable(batch_size) -> dict of tensors (the batch)
        model_fn: callable(**batch) -> output (forward pass)
        min_bs: minimum batch size to try
        max_bs: upper bound for search
        verbose: print progress

    Returns:
        int: the largest batch size that ran without OOM
    """
    def try_batch(bs):
        """Try a forward pass with the given batch size. Returns True if it fits."""
        torch.cuda.empty_cache()
        gc.collect()
        try:
            batch = make_batch_fn(bs)
            with torch.inference_mode():
                _ = model_fn(**batch)
                torch.cuda.synchronize()
            # Clean up
            del batch
            torch.cuda.empty_cache()
            gc.collect()
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                # Clean up after OOM
                torch.cuda.empty_cache()
                gc.collect()
                return False
            raise  # Re-raise non-OOM errors

    if verbose:
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"[GPU] {torch.cuda.get_device_name(0)}, {total_mem:.1f} GB VRAM", flush=True)
        print(f"[GPU] Searching for max batch size (range {min_bs}-{max_bs})...", flush=True)

    # First check if min_bs even fits
    if not try_batch(min_bs):
        if verbose:
            print(f"[GPU] ERROR: even batch_size={min_bs} doesn't fit!", flush=True)
        return min_bs

    # Quick exponential probe to find upper bound faster
    probe = min_bs
    while probe <= max_bs and try_batch(probe):
        last_good = probe
        probe *= 2
    # Now binary search between last_good and min(probe, max_bs)
    lo, hi = last_good, min(probe, max_bs)

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_batch(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Apply 90% safety margin for torch.compile overhead
    safe_bs = max(1, int(best * 0.9))

    if verbose:
        used = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[GPU] Max batch size that fits: {best}", flush=True)
        print(f"[GPU] Using safe batch size (90%): {safe_bs}", flush=True)
        print(f"[GPU] Memory: {used:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

    return safe_bs
