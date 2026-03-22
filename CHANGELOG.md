# changelog

## 0.3.1 — 2026-03
- focal loss rewritten in log-space (no more NaNs at |logit| > 30)
- smoothing vectorized via `sliding_window_view` (~8x faster on long series)
- bench script gained `--batch` and `--threads` knobs
- ci: added smoke-export job that builds an onnx model end-to-end
- synthetic sample data generator for dry runs

## 0.3.0 — 2026-02
- turbofan dataset wired in w/ RUL<30 cycle anomaly label
- int8 quant falls back to dynamic if static calibration fails
- ort thread pinning for raspberry pi
- docs: architecture + benchmarks + runbook

## 0.2.0 — 2026-01
- onnx export w/ dynamic batch + time axes
- mlflow tracker shim
- early stopping, checkpointing
- inference-only Dockerfile (no torch)

## 0.1.0 — 2025-12
- scaffolding, SKAB loader, sliding windows
- BiLSTM + Transformer hybrid model
- focal BCE loss, train/eval loop
