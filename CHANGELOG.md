# changelog

## 0.3.0 — 2026-02
- turbofan dataset wired in w/ RUL<30 cycle anomaly label
- int8 quant now falls back to dynamic if static calibration fails
- ort thread pinning for raspberry pi deployments
- docs: architecture + benchmarks

## 0.2.0 — 2026-01
- onnx export w/ dynamic batch + time axes
- mlflow tracker shim
- early stopping, checkpointing
- inference-only Dockerfile (no torch)

## 0.1.0 — 2025-12
- scaffolding, SKAB loader, sliding windows
- BiLSTM + Transformer hybrid model
- focal BCE loss, train/eval loop
