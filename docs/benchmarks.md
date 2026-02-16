# benchmarks

## latency (onnxruntime, CPU, 1-window batch)

| device            | fp32 p50 | int8 p50 | int8 p95 |
|-------------------|---------:|---------:|---------:|
| Ryzen 5 5600X     |   3.1 ms |   1.4 ms |   1.9 ms |
| Raspberry Pi 4B   |  112 ms  |  38 ms   |  46 ms   |
| Raspberry Pi 5    |   62 ms  |  21 ms   |  26 ms   |

target: <50ms per window on Pi 4B. int8 clears it with headroom.

## accuracy (val split, best-F1 threshold on val)

| dataset   | precision |  recall |      F1 |    AUROC |
|-----------|----------:|--------:|--------:|---------:|
| SKAB      |     0.91  |   0.88  |   0.89  |    0.97  |
| CMAPSS FD001 |  0.84  |   0.82  |   0.83  |    0.93  |

int8 drops F1 by ~0.8-1.1 pp relative to fp32 on both.
