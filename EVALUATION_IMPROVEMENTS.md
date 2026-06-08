# GraphMend Evaluation — Complete Results Across 25 Models

## Overview

This document presents the complete evaluation of GraphMend across 25 HuggingFace models spanning 9 architecture families. All results are from NVIDIA RTX 3090 (24GB) and A40 (48GB) GPUs, PyTorch 2.7, Transformers 4.52, CUDA 11.8.

**Headline results:** Up to **25x cold start speedup** (geomean 4.7x), consolidates up to **50 CUDA graphs into 1** (geomean 7x), up to **1.22x kernel fusion** (geomean 1.05x), and up to **1.34x warm forward pass speedup** (geomean 1.04x) across 24 models.

Results are consistent across both GPUs, confirming that graph break overhead is a **software-level bottleneck** in the TorchDynamo/Inductor compilation pipeline, independent of GPU hardware.

---

## 1. Complete Results (All Speedup Factors — 24 Models)

All values as Nx. Values >1.0x = improvement, <1.0x = regression.

| Model | Cold Start | Launches | Kernels | Warm Fwd | Throughput |
|---|---:|---:|---:|---:|---:|
| MoLFormer-XL | **25x** | 50→1 | 1.11x | 1.03x | 1.01x |
| Florence-2-large | **21x** | 30→1 | 1.02x | **1.13x** | **1.15x** |
| BART-Large-CNN | **21x** | 30→1 | 1.19x | **1.11x** | 1.02x |
| REBEL-Large | **20x** | 30→1 | 1.19x | **1.09x** | 1.00x |
| opus-mt-fr-en | **13x** | 17→1 | 1.07x | **1.10x** | — |
| BART-base | **12x** | 18→1 | 1.22x | 1.05x | 1.01x |
| layoutlmv3-base | **6.8x** | — | 1.05x | 1.00x | 0.99x |
| grounding-dino-tiny | **5.2x** | 79→14 | 1.11x | 1.01x | 0.98x |
| grounding-dino-base | **5.2x** | 79→14 | 1.10x | 0.99x | 0.98x |
| chronos-bolt-small | **4.6x** | 5→1 | 1.01x | 0.99x | 1.00x |
| t5-small | **3.5x** | 4→1 | 1.01x | 0.99x | 0.99x |
| flan-t5-large | **3.3x** | 4→1 | 1.00x | 0.96x | 1.06x |
| blenderbot-400M-distill | **3.2x** | 4→1 | 1.01x | 1.00x | 1.05x |
| Whisper-small | **3.1x** | 4→1 | 1.00x | 1.02x | 0.99x |
| Whisper-large-v3 | **3.1x** | 4→1 | 1.00x | 1.05x | 0.98x |
| inclusively-reform.-it5 | **3.0x** | 4→1 | 1.00x | 1.07x | — |
| tiny-random-PegasusForCausalLM | **2.7x** | 3→1 | 1.05x | **1.34x** | 1.05x |
| Whisper-base | **2.5x** | 4→1 | 1.00x | 0.99x | — |
| qwen-audio-chat | **2.4x** | 3→1 | 1.01x | 1.02x | 1.08x |
| Phi-4-mini-instruct | **2.3x** | 5→1 | 1.01x | 1.01x | 1.06x |
| t5-base | **2.3x** | 4→1 | 1.01x | 1.09x | 1.04x |
| longformer | **1.9x** | 14→4 | 1.07x | 1.00x | — |
| t5-3b | **1.8x** | 4→1 | 1.00x | 1.02x | 1.02x |
| BioGPT | **1.6x** | 3→1 | 1.01x | 1.01x | — |
| **GEOMEAN** | **4.7x** | **9→1** | **1.05x** | **1.04x** | **1.02x** |
| **MAX** | **25x** | **79→1** | **1.22x** | **1.34x** | **1.15x** |

---

## 2. Benchmark Suite (25 Models)

| Model | Architecture | Params | Graph Breaks | Break Reasons |
|---|---|---|---|---|
| biogpt | Causal LM | 150M | 2 | Dynamic control flow |
| blenderbot-400M-distill | Seq2Seq | 400M | 3 | Dynamic control flow + logger |
| flan-t5-large | Seq2Seq (T5) | 780M | 3 | Logger calls |
| longformer-base-4096 | Long-context | 148M | 5 | Logger + tensor.item() |
| moe-minicpm-x4-base | MoE | 4B | 15 | Dynamic shape operator |
| Phi-4-mini-instruct | Causal LM | 3.8B | 5 | Dynamic control flow |
| Qwen-Audio-Chat | Causal LM | 8B | 2 | Dynamic control flow |
| tiny-random-PegasusForCausalLM | Causal LM | 1M | 2 | Dynamic control flow + logger |
| t5-small | Seq2Seq (T5) | 60M | 3 | Logger calls |
| t5-base | Seq2Seq (T5) | 220M | 3 | Logger calls |
| t5-3b | Seq2Seq (T5) | 3B | 3 | Logger calls |
| bart-base | Seq2Seq (BART) | 140M | 7 | Tensor jump (isinf/isnan) + logger |
| bart-large-cnn | Seq2Seq (BART) | 400M | 7 | Tensor jump (isinf/isnan) + logger |
| rebel-large | Seq2Seq (BART) | 400M | 7 | Tensor jump (isinf/isnan) + logger |
| opus-mt-fr-en | Seq2Seq (Marian) | 74M | 6 | Tensor jump (isinf/isnan) + logger |
| inclusively-reformulation-it5 | Seq2Seq (IT5) | 3B | 3 | Logger calls |
| chronos-bolt-small | Time Series (T5) | 48M | 6 | Logger calls |
| whisper-small | Audio Seq2Seq | 244M | 3 | Logger calls |
| whisper-base | Audio Seq2Seq | 74M | 3 | Logger calls |
| whisper-large-v3 | Audio Seq2Seq | 1.5B | 3 | Logger calls |
| Florence-2-large | Vision-Language | 770M | 7 | Data-dep branching + generic jump |
| MoLFormer-XL-both-10pct | Molecular Encoder | 47M | 5 | Data-dep operator (torch.equal) |
| layoutlmv3-base | Document Understanding | 125M | 2 | Skipped function |
| grounding-dino-tiny | Object Detection | 172M | 17 | Dynamic shape + data-dep + tensor slicing |
| grounding-dino-base | Object Detection | 341M | 17 | Dynamic shape + data-dep + tensor slicing |

---

## 3. Graph Break Fix Rates

| Model | Breaks | Fixed (%) | Transformation |
|---|---|---|---|
| biogpt | 2 | 100% | Predicated Control Flow |
| blenderbot-400M-distill | 3 | 100% | Predicated Control Flow + Deferred Side Effects |
| flan-t5-large | 3 | 100% | Deferred Side Effects |
| Phi-4-mini-instruct | 5 | 100% | Predicated Control Flow |
| Qwen-Audio-Chat | 2 | 100% | Predicated Control Flow |
| tiny-random-PegasusForCausalLM | 2 | 100% | Predicated Control Flow + Deferred Side Effects |
| t5-small | 3 | 100% | Deferred Side Effects |
| t5-base | 3 | 100% | Deferred Side Effects |
| t5-3b | 3 | 100% | Deferred Side Effects |
| bart-base | 7 | 100% | Predicated Control Flow + Deferred Side Effects |
| bart-large-cnn | 7 | 100% | Predicated Control Flow + Deferred Side Effects |
| rebel-large | 7 | 100% | Predicated Control Flow + Deferred Side Effects |
| opus-mt-fr-en | 6 | 100% | Predicated Control Flow + Deferred Side Effects |
| inclusively-reformulation-it5 | 3 | 100% | Deferred Side Effects |
| chronos-bolt-small | 6 | 100% | Deferred Side Effects |
| whisper-small | 3 | 100% | Deferred Side Effects |
| whisper-base | 3 | 100% | Deferred Side Effects |
| whisper-large-v3 | 3 | 100% | Deferred Side Effects |
| Florence-2-large | 7 | 100% | Predicated Control Flow |
| MoLFormer-XL-both-10pct | 5 | 100% | Predicated Guards |
| layoutlmv3-base | 2 | 100% | Deferred Side Effects |
| longformer-base-4096 | 5 | 40% | Partial (tensor.item() unfixable) |
| grounding-dino-tiny | 17 | 58% | Partial (dynamic shapes unfixable) |
| grounding-dino-base | 17 | 58% | Partial (dynamic shapes unfixable) |
| moe-minicpm-x4-base | 15 | 0% | Unfixable (dynamic shape operators) |

**Summary: 21/25 fully fixed (84%), 3 partially fixed, 1 unfixable**

---

## 4. Three Transformations

### Transformation 1: Predicated Dynamic Control Flow (`torch.where`)

**Pattern:** Data-dependent branching on tensor values
```python
# ORIGINAL (graph break — forces D2H sync):
if hidden_states.dtype == torch.float16 and (
    torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
):
    hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

# FIXED (no graph break — stays in graph):
if hidden_states.dtype == torch.float16:
    needs_clamp = torch.isinf(hidden_states) | torch.isnan(hidden_states)
    hidden_states = torch.where(needs_clamp, torch.clamp(hidden_states, ...), hidden_states)
```

**Applied to:** biogpt, blenderbot, Phi-4-mini, Qwen-Audio-Chat, Pegasus, bart-base, bart-large-cnn, rebel-large, opus-mt-fr-en, Florence-2-large

### Transformation 2: Deferred Side Effects (Logger removal)

**Pattern:** `logger.warning_once(...)` in forward path
```python
# ORIGINAL (graph break):
logger.warning_once("Passing a tuple of past_key_values is deprecated...")

# FIXED: removed (deprecation warning is a side effect with no computational impact)
```

**Applied to:** flan-t5-large, t5-small/base/3b, bart-base/large-cnn, rebel-large, opus-mt-fr-en, inclusively-reformulation-it5, chronos-bolt-small, whisper-small/base/large-v3, layoutlmv3-base

### Transformation 3: Predicated Guards (`torch._assert_async`) — NEW

**Pattern:** Validation guard with data-dependent operator
```python
# ORIGINAL (graph break — torch.equal forces D2H):
if not torch.equal(attention_mask, per_query_extended):
    raise ValueError("does not support arbitrary 3D attention")

# FIXED (no graph break — graph-native assertion):
torch._assert_async(
    torch.all(attention_mask == per_query_extended),
    "does not support arbitrary 3D attention"
)
```

**Applied to:** MoLFormer-XL-both-10pct

**Note:** `torch._check` does NOT work here — it requires a Python `bool`, not a `FakeTensor` during Dynamo tracing. Only `torch._assert_async` accepts tensor inputs directly.

---

## 5. Cold Start Compilation Speedup

| Model | Original | Fixed | Speedup |
|---|---:|---:|---:|
| MoLFormer-XL | 6201ms | 251ms | **25x** |
| BART-Large-CNN | 3915ms | 186ms | **21x** |
| Florence-2-large | 4813ms | 230ms | **21x** |
| REBEL-Large | 3784ms | 191ms | **20x** |
| opus-mt-fr-en | 2059ms | 156ms | **13x** |
| BART-base | 2238ms | 189ms | **12x** |
| layoutlmv3-base | 331ms | 49ms | **6.8x** |
| grounding-dino-tiny | 14678ms | 2822ms | **5.2x** |
| grounding-dino-base | 15569ms | 3009ms | **5.2x** |
| chronos-bolt-small | 823ms | 177ms | **4.6x** |
| t5-small | 514ms | 148ms | **3.5x** |
| flan-t5-large | 1428ms | 437ms | **3.3x** |
| blenderbot-400M-distill | 392ms | 122ms | **3.2x** |
| Whisper-small | 619ms | 200ms | **3.1x** |
| Whisper-large-v3 | 766ms | 250ms | **3.1x** |
| inclusively-reform.-it5 | 720ms | 239ms | **3.0x** |
| tiny-random-PegasusForCausalLM | 295ms | 108ms | **2.7x** |
| Whisper-base | 650ms | 261ms | **2.5x** |
| Qwen-Audio-Chat | 658ms | 275ms | **2.4x** |
| t5-base | 584ms | 257ms | **2.3x** |
| Phi-4-mini-instruct | 633ms | 279ms | **2.3x** |
| longformer | 1872ms | 967ms | **1.9x** |
| t5-3b | 729ms | 404ms | **1.8x** |
| BioGPT | 331ms | 203ms | **1.6x** |

**Geomean: 4.7x   Median: 3.2x   Max: 25x**

---

## 6. CUDA Graph Consolidation

| Model | Original Launches | Fixed Launches | Consolidation |
|---|---:|---:|---:|
| grounding-dino-base | 79 | 14 | **6x** |
| grounding-dino-tiny | 79 | 14 | **6x** |
| MoLFormer-XL | 50 | 1 | **50x** |
| Florence-2-large | 30 | 1 | **30x** |
| BART-Large-CNN | 30 | 1 | **30x** |
| REBEL-Large | 30 | 1 | **30x** |
| BART-base | 18 | 1 | **18x** |
| opus-mt-fr-en | 17 | 1 | **17x** |
| longformer | 14 | 4 | **4x** |
| chronos-bolt-small | 5 | 1 | **5x** |
| Phi-4-mini-instruct | 5 | 1 | **5x** |
| blenderbot-400M-distill | 4 | 1 | **4x** |
| flan-t5-large | 4 | 1 | **4x** |
| inclusively-reform.-it5 | 4 | 1 | **4x** |
| t5-small | 4 | 1 | **4x** |
| t5-base | 4 | 1 | **4x** |
| t5-3b | 4 | 1 | **4x** |
| Whisper-small | 4 | 1 | **4x** |
| Whisper-base | 4 | 1 | **4x** |
| Whisper-large-v3 | 4 | 1 | **4x** |
| BioGPT | 3 | 1 | **3x** |
| Qwen-Audio-Chat | 3 | 1 | **3x** |
| tiny-random-PegasusForCausalLM | 3 | 1 | **3x** |

**Geomean: 7x   Max: 50x**

---

## 7. Kernel Fusion

| Model | Original | Fixed | Fusion |
|---|---:|---:|---:|
| BART-base | 245 | 201 | **1.22x** (+44) |
| BART-Large-CNN | 486 | 408 | **1.19x** (+78) |
| REBEL-Large | 486 | 408 | **1.19x** (+78) |
| grounding-dino-tiny | 1540 | 1383 | **1.11x** (+157) |
| MoLFormer-XL | 283 | 255 | **1.11x** (+28) |
| grounding-dino-base | 1724 | 1567 | **1.10x** (+157) |
| opus-mt-fr-en | 295 | 275 | **1.07x** (+20) |
| longformer | 311 | 290 | **1.07x** (+21) |
| layoutlmv3-base | 211 | 200 | **1.05x** (+11) |
| tiny-random-PegasusForCausalLM | 39 | 37 | **1.05x** (+2) |
| Florence-2-large | 1003 | 980 | **1.02x** (+23) |
| Phi-4-mini-instruct | 469 | 463 | **1.01x** (+6) |
| t5-small | 254 | 251 | **1.01x** (+3) |
| chronos-bolt-small | 314 | 311 | **1.01x** (+3) |
| blenderbot-400M-distill | 330 | 327 | **1.01x** (+3) |
| Qwen-Audio-Chat | 334 | 331 | **1.01x** (+3) |
| BioGPT | 318 | 316 | **1.01x** (+2) |
| t5-base | 494 | 491 | **1.01x** (+3) |
| Whisper-base | 219 | 218 | **1.00x** (+1) |
| t5-3b | 1070 | 1067 | **1.00x** (+3) |
| Whisper-small | 471 | 470 | **1.00x** (+1) |
| inclusively-reform.-it5 | 1045 | 1043 | **1.00x** (+2) |
| flan-t5-large | 1070 | 1068 | **1.00x** (+2) |
| Whisper-large-v3 | 1137 | 1136 | **1.00x** (+1) |

**Geomean: 1.05x   Max: 1.22x**

---

## 8. Warm Forward Pass Speedup

| Model | Original | Fixed | Speedup |
|---|---:|---:|---:|
| tiny-random-PegasusForCausalLM | 0.7ms | 0.5ms | **1.34x** |
| Florence-2-large | 48.2ms | 42.7ms | **1.13x** |
| BART-Large-CNN | 53.6ms | 48.3ms | **1.11x** |
| opus-mt-fr-en | 29.3ms | 26.6ms | **1.10x** |
| REBEL-Large | 47.4ms | 43.5ms | **1.09x** |
| t5-base | 27.8ms | 25.6ms | **1.09x** |
| inclusively-reform.-it5 | 65.1ms | 60.9ms | **1.07x** |
| BART-base | 72.3ms | 68.6ms | **1.05x** |
| Whisper-large-v3 | 53.5ms | 50.9ms | **1.05x** |
| MoLFormer-XL | 120.2ms | 116.2ms | **1.03x** |
| t5-3b | 74.3ms | 72.6ms | **1.02x** |
| Qwen-Audio-Chat | 68.2ms | 66.7ms | **1.02x** |
| Whisper-small | 66.2ms | 65.1ms | **1.02x** |
| Phi-4-mini-instruct | 72.1ms | 71.4ms | **1.01x** |
| grounding-dino-tiny | 93.7ms | 93.1ms | **1.01x** |
| BioGPT | 23.8ms | 23.7ms | **1.01x** |
| longformer | 541.9ms | 540.1ms | **1.00x** |
| layoutlmv3-base | 48.2ms | 48.1ms | **1.00x** |
| blenderbot-400M-distill | 2.3ms | 2.3ms | **1.00x** |
| t5-small | 22.9ms | 23.1ms | **0.99x** |
| chronos-bolt-small | 2.6ms | 2.7ms | **0.99x** |
| grounding-dino-base | 118.8ms | 120.0ms | **0.99x** |
| Whisper-base | 137.9ms | 139.5ms | **0.99x** |
| flan-t5-large | 34.2ms | 35.5ms | **0.96x** |

**Geomean: 1.04x   Max: 1.34x**

---

## 9. Throughput Speedup

| Model | Original | Fixed | Speedup |
|---|---:|---:|---:|
| Florence-2-large | 69.2 | 79.3 | **1.15x** |
| Qwen-Audio-Chat | 22.8 | 24.6 | **1.08x** |
| flan-t5-large | 30.9 | 32.7 | **1.06x** |
| Phi-4-mini-instruct | 27.3 | 28.9 | **1.06x** |
| blenderbot-400M-distill | 37.0 | 38.7 | **1.05x** |
| tiny-random-PegasusForCausalLM | 47.9 | 50.4 | **1.05x** |
| t5-base | 63.6 | 66.2 | **1.04x** |
| BART-Large-CNN | 9.7 | 9.9 | **1.02x** |
| t5-3b | 34.2 | 34.8 | **1.02x** |
| BART-base | 2.8 | 2.8 | **1.01x** |
| MoLFormer-XL | 7081.6 | 7159.7 | **1.01x** |
| chronos-bolt-small | 5162.5 | 5172.2 | **1.00x** |
| REBEL-Large | 16.2 | 16.2 | **1.00x** |
| layoutlmv3-base | 647.0 | 643.5 | **0.99x** |
| Whisper-small | 99.7 | 98.9 | **0.99x** |
| t5-small | 99.4 | 98.3 | **0.99x** |
| grounding-dino-base | 8.8 | 8.7 | **0.98x** |
| grounding-dino-tiny | 11.6 | 11.3 | **0.98x** |
| Whisper-large-v3 | 23.8 | 23.3 | **0.98x** |

**Geomean: 1.02x   Max: 1.15x**

---

## 10. Cross-Device Consistency (A40 vs RTX 3090)

All models evaluated on both NVIDIA A40 (48GB, data center) and RTX 3090 (24GB, consumer).

- **Cold start speedup** is device-independent (within 2-5% across devices). Cold start overhead is CPU-side compilation, not GPU compute.
- **Warm speedup** follows the same pattern on both GPUs. The break type determines improvement, not the GPU hardware.
- **Throughput** rankings are preserved across devices.

Graph break overhead is a **software bottleneck** in the PyTorch 2 compilation pipeline. GraphMend's AST-level transformations are portable across any CUDA GPU.

---

## 11. Summary

| Metric | Geomean | Max |
|---|---:|---:|
| **Cold start speedup** | **4.7x** | **25x** |
| **CUDA graph consolidation** | **7x** | **50x** |
| **Kernel fusion** | **1.05x** | **1.22x** |
| **Warm forward pass** | **1.04x** | **1.34x** |
| **Throughput** | **1.02x** | **1.15x** |
| Fix rate | 84% (21/25) | 100% |

---

## 12. Key Findings

1. **All graph breaks hurt cold start.** GraphMend achieves up to **25x cold start speedup** (geomean 4.7x) by consolidating separate compilation units into one.

2. **Kernel fusion drives warm improvement.** Eliminating breaks allows TorchInductor to fuse kernels across former break boundaries, achieving up to **1.22x fewer kernels** and up to **1.34x warm speedup** (geomean 1.04x).

3. **CUDA graph consolidation up to 50x.** Original models have 3-50 separate CUDA graph launches per forward pass. Fixed models have 1 (partially fixed models like grounding-dino go from 79→14).

4. **Results are hardware-independent.** A40 and RTX 3090 show consistent patterns, confirming software-level bottleneck.

5. **Three transformations cover all fixable patterns.** Predicated Control Flow (torch.where), Deferred Side Effects (logger removal), and Predicated Guards (torch._assert_async) fix 84% of models completely.

---

## 13. Files Reference

| File | Description |
|---|---|
| `paper_charts/B1_cold_start_all.*` | Cold start — all models |
| `paper_charts/B2_warm_data_dep.*` | Warm latency — data-dep models |
| `paper_charts/B3_throughput_data_dep.*` | Throughput — data-dep models |
| `paper_charts/B4_break_type_comparison.*` | Break type impact comparison |
| `paper_charts/B5_break_counts.*` | Breaks fixed vs remaining |
| `paper_charts/S1_subgraphs_vs_cold.*` | Scatter: breaks vs cold improvement |
| `paper_charts/S2_size_vs_warm.*` | Scatter: model size vs warm improvement |
| `paper_charts/H1_heatmap.*` | Heatmap: all models × all metrics |
| `paper_charts/D1_fix_rate_pie.*` | Fix rate distribution |
| `paper_charts/D2_break_type_dist.*` | Break type distribution |
| `pytorch diagrams/Graphs/generate_all_paper_charts.py` | Chart generation script |
| `pytorch diagrams/Graphs/3090/` | RTX 3090 trace JSONs |
| `pytorch diagrams/Graphs/A40_new/` | A40 trace JSONs |
| `models/*/traces/*.log` | Benchmark logs |
| `transformers-modified/src/transformers/models/bart/modeling_bart.py` | BART fixes |
| `transformers-modified/src/transformers/models/whisper/modeling_whisper.py` | Whisper fixes |
| `fixed_model_files/modeling_molformer.py` | MoLFormer fixes |
| `fixed_model_files/modeling_florence2.py` | Florence-2 fixes |
