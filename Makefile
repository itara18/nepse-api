.PHONY: fetch install test lint format run load train research
install:
	pip install -r requirements-dev.txt
test:
	pytest --cov=app --cov-report=term-missing
lint:
	ruff check .
format:
	ruff check . --fix
run:
	uvicorn app.main:app --reload
load:
	python -m app.loader data/
train:
	python -m app.training
fetch:
	python -m app.fetcher

research:
	python -m app.research data results
