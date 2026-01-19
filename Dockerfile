FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt requirements-infer.txt ./
RUN pip install -r requirements-infer.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["python", "-m"]
CMD ["scripts.bench", "--model", "/models/model.int8.onnx"]
