# architecture

```
raw csv ──► Standardizer ──► sliding windows ──► WindowedSeries
                                                     │
                                                     ▼
                          ┌─────── BiLSTMEncoder (local temporal) ───────┐
                          │                                              │
                          └── Transformer stack (long-range attention) ──┘
                                                     │
                                                     ▼
                                             last-step head ──► sigmoid
                                                     │
                                           threshold + median smooth
```

## why hybrid

- BiLSTM is great at local slope/step features on short horizons.
- Self-attention catches cross-time correlations that fall outside the LSTM's effective memory.
- Stack of 2+2 stays small enough to run <50ms int8 on a Pi 4.

## why int8 QDQ

- QDQ keeps the graph portable across ORT execution providers.
- Per-channel weight quant holds accuracy within ~1pp F1 vs fp32 on SKAB.
