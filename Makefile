.PHONY: install dev test lint format docker-up docker-down seed eval

install:
	pip install -e ".[dev]"

dev:
	uvicorn backend.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

docker-up:
	docker compose -f infra/docker-compose.yml up --build

docker-down:
	docker compose -f infra/docker-compose.yml down

seed:
	python scripts/seed.py

eval:
	python eval/rag_eval.py
