# ts-anomaly

![ci](https://github.com/shrizon/ts-anomaly/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

industrial multi-sensor time-series anomaly detection. **BiLSTM + Transformer** hybrid, trained on **SKAB** and **NASA C-MAPSS**, shipped as an **INT8 ONNX** model that runs in **<50ms per window on a Raspberry Pi 4B**.

```
raw csv  →  standardize  →  sliding windows  →  BiLSTM  →  Transformer  →  sigmoid
                                                                             │
                                                                threshold + median smooth
```

## results (val split, best-F1 threshold)

| dataset      |     F1 |  AUROC |  int8 Δ-F1 |
|--------------|-------:|-------:|-----------:|
| SKAB         |  0.89  |  0.97  |    −0.8 pp |
| CMAPSS FD001 |  0.83  |  0.93  |    −1.1 pp |

| device          | int8 p50 | int8 p95 |
|-----------------|---------:|---------:|
| Ryzen 5 5600X   |   1.4 ms |   1.9 ms |
| Raspberry Pi 5  |    21 ms |    26 ms |
| Raspberry Pi 4B |    38 ms |    46 ms |

## what's in the box

- PyTorch model (`src/models/hybrid.py`) — BiLSTM encoder → small Transformer stack → last/mean pool → sigmoid
- Focal BCE loss (log-space stable) so the rare-anomaly tail isn't ignored
- ONNX export w/ dynamic batch & time, static INT8 QDQ w/ dynamic fallback
- MLflow-lite tracker — no-op if mlflow isn't installed
- Slim inference Dockerfile: numpy + onnxruntime, no torch
- Grid sweep runner, early stopping, per-channel standardization, median post-smooth

## quickstart

```bash
make install
./scripts/fetch_data.sh            # or: python scripts/make_sample.py
make train
python scripts/export.py --ckpt checkpoints/last.pt
make bench
```

see [`docs/architecture.md`](docs/architecture.md), [`docs/benchmarks.md`](docs/benchmarks.md), and [`docs/runbook.md`](docs/runbook.md).

## datasets

- **SKAB** — water-circulation rig w/ labelled anomalies. <https://github.com/waico/SKAB>
- **NASA C-MAPSS** — turbofan engine degradation. RUL < 30 cycles → anomaly label.

## layout

```
src/
  data/       # loaders, windows, scaling, dataset
  models/     # bilstm, transformer, hybrid, loss
  train/      # loop, early stop, checkpoint
  eval/       # metrics, threshold, report
  export/     # onnx + int8 quant
  infer/      # ort session, batched scoring, smoothing
scripts/      # train, export, bench, infer, eval, sweep
configs/      # default.yaml, turbofan.yaml, sweep.yaml
docs/         # architecture, benchmarks, runbook
```

## license

MIT.
