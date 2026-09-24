# syntax=docker/dockerfile:1.7

# ---- Stage 1: fetch pinned upstream artifacts and convert MiniFASNet to ONNX ----------
# torch is only present in this throwaway stage; it never reaches the runtime image.
FROM python:3.13-slim AS models
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY tools/requirements-convert.txt tools/fetch_models.sh tools/convert_minifasnet.py \
     tools/model_digests.sha256 tools/
RUN pip install --no-cache-dir -r tools/requirements-convert.txt
RUN bash tools/fetch_models.sh /build/models \
 && python tools/convert_minifasnet.py --model-dir /build/models \
 && cd /build/models && sha256sum -c /build/tools/model_digests.sha256

# ---- Stage 2: runtime --------------------------------------------------------------------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LIVENESS_MODEL_DIR=/models \
    LIVENESS_PORT=8080
RUN groupadd --system --gid 10001 liveness \
 && useradd --system --uid 10001 --gid liveness --no-create-home --shell /usr/sbin/nologin liveness
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && rm -rf /app/src /app/build
COPY --from=models /build/models/*.onnx /models/
COPY --from=models /build/models/manifest.json /models/manifest.json
COPY --from=models /build/models/upstream/LICENSE.* /models/licenses/
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2).status == 200 else 1)"]
CMD ["face-liveness"]
