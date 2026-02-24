# System Architecture & Engineering Defense

## 1. Model Selection Defense
Chosen model: `Qwen2.5-VL-2B`.

Rationale: this challenge requires multi-frame temporal reasoning under free-tier GPU constraints. On Kaggle T4, the practical risk is activation memory growth from video tokens, not just static parameter storage. A 2B base model in 4-bit leaves enough headroom for 8-frame clips, LoRA adapters, optimizer states, and checkpointing overhead.

VRAM fit comparison (target: Kaggle 2xT4, 16 GB each):

| Model | Params | 4-bit Base (approx) | Video Fine-Tune VRAM (approx) | Fit on T4 for 8-frame QLoRA |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-2B | 2B | 2-3 GB | ~10-12 GB | Yes |
| LLaVA-NeXT-Video-7B | 7B | 8-10 GB | ~18-24 GB | High OOM risk |
| VideoLLaMA2-7B | 7B | 9-11 GB | ~20-26 GB | High OOM risk |

Engineering decision: prioritize stable training completion and reproducibility over larger-parameter baselines likely to fail in free-tier sessions.

## 2. Frame Sampling Rationale
Sampling strategy: Motion-Magnitude Adaptive Sampling using Farneback dense optical flow.

Uniform sampling was explicitly avoided. In packaging workflows, critical boundary transitions often happen in short bursts (for example, tape tear, label placement, handoff posture change) surrounded by low-motion periods. Uniform picks over-represent static frames and under-sample boundary frames.

Implemented approach in `data_pipeline.py`:
- Compute dense optical flow magnitude between consecutive frames inside annotated temporal windows.
- Rank frames by mean magnitude (motion energy).
- Select top-8 motion frames and then sort chronologically before VLM ingestion.
- Fallback to short-window sequential indices when boundary windows are too small.

Boundary coverage sketch:

```text
Timeline:  |--------- Box Setup ---------|-- Tape --|---- Put Items ----|
Motion:    low  low  med  HIGH HIGH  med  HIGH HIGH  med  low  low  med
Uniform:   ^         ^         ^         ^         ^         ^
Adaptive:                ^   ^      ^   ^      ^        ^
                          (focus near temporal change points)
```

This increases semantic density per frame token and directly supports tIoU and AA@1 improvements under the same VRAM budget.

## 3. Failure Mode Analysis
Most frequent confusion: `Tape` vs `Pack`.

Observed pattern: both classes share similar scene context (same workstation, same box geometry, similar hand trajectories). When tape handling starts or ends near clip edges, frame evidence can look nearly identical to generic packing motions.

Likely causes:
- Boundary ambiguity: start/end of tape action may occur between sampled frames.
- Occlusion: hands and tools partially hide the contact point with the box seam.
- Temporal truncation: 5-second windows sometimes miss full pre/post context needed for disambiguation.

Mitigation options (within constraints):
- Keep 8-frame budget but bias adaptive sampler to include both high-motion peaks and one low-motion context frame before/after each peak cluster.
- Add class-balanced loss weighting for `Tape` and `Pack` transitions.
- Expand anticipation supervision with transition priors learned from operation bigrams.
