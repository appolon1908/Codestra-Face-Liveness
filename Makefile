PY ?= .venv/bin/python
MODEL_DIR ?= models

.PHONY: venv models test lint typecheck check openapi run docker compose-up

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install --no-deps -e .

models:  ## fetch pinned artifacts + convert (needs torch: pip install -r tools/requirements-convert.txt)
	bash tools/fetch_models.sh $(MODEL_DIR)
	$(PY) tools/convert_minifasnet.py --model-dir $(MODEL_DIR)
	cd $(MODEL_DIR) && sha256sum -c ../tools/model_digests.sha256

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests tools
	$(PY) -m ruff format --check src tests tools

typecheck:
	$(PY) -m mypy

check: lint typecheck test

openapi:
	$(PY) tools/export_openapi.py

run:
	LIVENESS_MODEL_DIR=$(MODEL_DIR) $(PY) -m face_liveness.main

docker:
	docker build -t codestra/face-liveness:dev .

compose-up:
	@test -f secrets/liveness_api_token || (mkdir -p secrets && openssl rand -hex 32 > secrets/liveness_api_token)
	docker compose up --build
