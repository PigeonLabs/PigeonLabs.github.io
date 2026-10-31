# Realtime (browser) vs. Python online detector

Measured 2026-11-01 on Apple M1 Max, Chromium WebGPU, ONNX Runtime Web 1.29.0.
Reference: `data/outputs/*.json` (Python online detector, PyTorch MPS FP32 features).
Agreement = share of frames where both sides make the same alarm decision under the video's criterion.

## Per-observation time (realtime mode)

| Step | FP32 graph | FP16 graph |
| --- | ---: | ---: |
| JPEG decode + pixels | 1.1 ms | 1.1 ms |
| PIL-compatible bicubic resize + normalize (JS) | 8.5 ms | 8.5 ms |
| DINOv2-base + pooling (WebGPU) | 70.2 ms | 44.4 ms |
| Tracker + A + C (JS) | ~1 ms | ~1 ms |

With 2-frame stride, FP16 keeps up with 30 fps playback; FP32 reaches about 24 fps.
In the page (pipeline in a Web Worker, UI drawing on the main thread) FP16 measured 53–58 ms per
observation end to end and held 30.0 fps.

## Alarm agreement

| Video | Criterion | FP16 agreement | FP16 alarm frames (JS vs Py) | FP16 max score diff |
| --- | --- | ---: | ---: | ---: |
| 리프트 정상 | paper | 100.00% | 0 vs 0 | 1.023 |
| 리프트 이상 | q95 | 100.00% | 296 vs 296 | 0.689 |
| 지게차 정상 | paper | 100.00% | 0 vs 0 | 0.641 |
| 지게차 이상 | q95 | 97.53% | 169 vs 173 | 4.877 |
| 절단 정상 | paper | 100.00% | 0 vs 0 | 0.693 |
| 절단 이상 | q99 | 99.57% | 374 vs 376 | 0.676 |

FP32 graph (WebGPU, 절단 이상): agreement 100% under all three criteria, max score diff 0.026.
FP32 graph (onnxruntime-node CPU + engine.js on PIL-decoded frames, 절단 이상 and 리프트 이상):
agreement 100% under all three criteria, max score diff ≤ 0.026; JS resize identical to PIL.
