# contributing

keep it small. prefer:

- one change per PR. if you're tempted to say "and also", open a second PR.
- tests that fail without your change and pass with it
- no new top-level deps w/o a short note in the PR on why

run locally:

```bash
make install
pre-commit install
make test
```

if you touch the model graph or input signature, regenerate the onnx fixture and re-run `scripts/bench.py` so the numbers in `docs/benchmarks.md` stay honest.
