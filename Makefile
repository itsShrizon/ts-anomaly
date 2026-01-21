PY ?= python

.PHONY: install test train export bench docker fmt

install:
	$(PY) -m pip install -r requirements.txt -r requirements-infer.txt -r requirements-dev.txt

test:
	$(PY) -m pytest -q

train:
	$(PY) scripts/train.py --config configs/default.yaml --out checkpoints/last.pt

export:
	$(PY) scripts/export.py --ckpt checkpoints/last.pt

bench:
	$(PY) scripts/bench.py --model artifacts/model.int8.onnx

docker:
	docker build -t ts-anomaly:infer .

fmt:
	ruff check --fix src tests scripts
	ruff format src tests scripts
