# ts-anomaly

industrial multi-sensor time-series anomaly detection. BiLSTM + Transformer hybrid, trained on **SKAB** and **NASA C-MAPSS**, shipped as an **INT8 ONNX** model that runs in <50ms per window on a Raspberry Pi 4B.

```
raw csv  →  standardize  →  sliding windows  →  BiLSTM  →  Transformer  →  sigmoid
                                                                             │
                                                                threshold + median smooth
```

## what's in the box

- PyTorch model (`src/models/hybrid.py`) — BiLSTM encoder feeding a small Transformer stack
- Focal BCE loss so the rare-anomaly tail doesn't get ignored
- ORT export + static INT8 quantization w/ dynamic fallback
- MLflow-lite tracker (no-op if mlflow isn't installed)
- Slim inference Dockerfile (numpy + onnxruntime only, no torch)
- Grid sweep runner, early stop, per-channel standardization

## quickstart

```bash
make install
./scripts/fetch_data.sh
make train
python scripts/export.py --ckpt checkpoints/last.pt
make bench
```

see `docs/architecture.md` and `docs/benchmarks.md` for details.

## datasets

- **SKAB** — multivariate water-circulation rig, labelled anomalies. <https://github.com/waico/SKAB>
- **NASA C-MAPSS** — turbofan engine degradation. we convert RUL < 30 cycles → anomaly.

## layout

```
src/
  data/       # loaders, windows, scaling, dataset
  models/     # bilstm, transformer, hybrid, loss
  train/      # loop, early stop, checkpoint
  eval/       # metrics, threshold tuning
  export/     # onnx + int8 quant
  infer/      # ort session, batched scoring, smoothing
scripts/      # train, export, bench, infer, sweep
configs/      # default.yaml, turbofan.yaml, sweep.yaml
```

## license

MIT.
