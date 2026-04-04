# runbook

## retraining

1. `./scripts/fetch_data.sh` — or `python scripts/make_sample.py` for smoke tests
2. pick a config: `configs/default.yaml` (SKAB) or `configs/turbofan.yaml`
3. `make train` — writes `checkpoints/last.pt`
4. `python scripts/eval.py --model artifacts/model.onnx --x val_x.npy --y val_y.npy`
5. `python scripts/export.py --ckpt checkpoints/last.pt` — writes `model.onnx` + `model.int8.onnx`
6. ship `artifacts/model.int8.onnx` alongside `threshold.json`

## pi deployment

```bash
docker build -t ts-anomaly:infer .
docker run --rm -v $PWD/artifacts:/models ts-anomaly:infer \
  scripts.bench --model /models/model.int8.onnx --threads 2
```

thread tuning:
- Pi 4B: `--threads 2` (4 cores, cache-bound; more threads → worse p95)
- Pi 5: `--threads 4`
- x86 desktop: usually `--threads 1` for single-window latency, more for batch throughput

## when F1 drops in prod

1. compare score histogram vs training-time histogram — if mean/std shifted, it's drift
2. check threshold file version vs model version — they must ship as a pair
3. fp32 vs int8 gap > 2pp F1 → calibration set isn't representative; re-run w/ `--calib 500`
4. if only recall dropped, try reducing `median_smooth` `k` — smoothing can eat short anomalies

## paging

`src/track.py` is a no-op shim by default. to enable MLflow:

```bash
pip install mlflow
export MLFLOW_TRACKING_URI=http://mlflow:5000
```

`docker compose up mlflow` spins one up locally on port 5000.
