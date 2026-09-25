FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install -r server/requirements.txt

COPY engine/counter.py engine/counter.py
COPY engine/models/01_XMG_s_1280.onnx engine/models/01_XMG_s_1280.onnx
COPY server/main.py server/main.py

ENV PORT=8080 ORT_THREADS=2
WORKDIR /app/server
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
