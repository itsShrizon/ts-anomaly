# runbook

## retraining

1. `./scripts/fetch_data.sh`
2. pick a config: `configs/default.yaml` (SKAB) or `configs/turbofan.yaml`
3. `make train` — writes `checkpoints/last.pt`
4. tune threshold on validation: produced implicitly by `scripts/eval.py`
5. export: `python scripts/export.py --ckpt checkpoints/last.pt`
6. ship `artifacts/model.int8.onnx` + `threshold.json`

## pi deployment

```bash
docker build -t ts-anomaly:infer .
docker run --rm -v $PWD/artifacts:/models ts-anomaly:infer \
  scripts.bench --model /models/model.int8.onnx --threads 2
```

pin threads to 2 on Pi 4 / 4 on Pi 5. more threads ≠ better here; cache contention wins.

## when F1 drops

- check threshold file version vs. model version — freeze them together
- rerun calibration with more samples (`--calib 500`)
- fp32 vs int8 gap > 2pp F1 → suspect calibration set isn't representative
