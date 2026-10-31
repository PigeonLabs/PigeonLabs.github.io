# CycleVAD demo

Browser demo of CycleVAD's **A+C** online detector on three IPAD real-world processes
(lift, forklift, cutting): one held-out normal video and one anomaly video each.
Research code: [PigeonLabs/KNU_Capstone1_CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD).

## Two modes (top-bar toggle)

| Mode | What runs |
| --- | --- |
| 사전 계산 (preprocessed) | Replays outputs of the Python online detector, recorded frame by frame from the JPEGs on an Apple M1 Max |
| 실시간 (realtime) | Runs in your browser: DINOv2-base via ONNX Runtime Web on **WebGPU** (FP16 when the GPU supports `shader-f16`, otherwise FP32; WASM if WebGPU is unavailable), then the causal cycle tracker and A+C scoring in JavaScript (`js/engine.js`). The model (~173 MB) is downloaded once and cached |

Both modes process frames strictly in order; no future frame is used for a decision.
Observations are made every 2nd frame and held in between, as in the paper.

Checks against the Python detector (same videos):

- JS bicubic resize is pixel-identical to PIL
- FP32 graph: alarm agreement 100%, max score difference 0.026
- FP16 graph: see `docs/parity.md`

## Thresholds

The three threshold values (paper rule: 99th percentile of 0.2T-window maxima; frame-score 99th
and 95th percentiles) are computed from normal videos only. **Which one is applied to each demo
video was chosen afterwards by comparing against that video's ground truth**, so the per-video
numbers shown are a best case for presentation, not a generalizable result.

## Data and models

- Frames are from the IPAD dataset: J. Liu et al., "IPAD: Industrial Process Anomaly Detection
  Dataset", IEEE TCSVT 2025 ([project page](https://ljf1113.github.io/IPAD_VAD/),
  [arXiv:2404.15033](https://arxiv.org/abs/2404.15033)). All rights to the frames remain with
  the dataset authors.
- Scene models were fitted on normal training videos only (fold 0 splits, seed 42).
- DINOv2-base (facebook/dinov2-base) weights are exported to ONNX, stored as FP16.
