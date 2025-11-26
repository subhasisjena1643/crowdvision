# CrowdVision Makefile

.PHONY: help setup install test clean run-api run-mlflow

help:
	@echo "CrowdVision Development Commands"
	@echo "================================="
	@echo "make setup       - Set up development environment"
	@echo "make install     - Install dependencies"
	@echo "make test        - Run tests"
	@echo "make test-unit   - Run unit tests only"
	@echo "make test-prop   - Run property tests only"
	@echo "make clean       - Clean temporary files"
	@echo "make run-api     - Start FastAPI server"
	@echo "make run-mlflow  - Start MLflow server"
	@echo "make init-mlflow - Initialize MLflow experiments"

setup:
	python scripts/setup_env.py

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m unit

test-prop:
	pytest tests/ -v -m property

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-mlflow:
	mlflow server --host 0.0.0.0 --port 5000

init-mlflow:
	python scripts/init_mlflow.py
