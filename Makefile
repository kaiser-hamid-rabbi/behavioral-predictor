.PHONY: help install dev test lint run worker docker-up docker-down train load-data features

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	cd backend && pip install -e .

dev: ## Install development dependencies
	cd backend && pip install -e ".[dev]"

test: ## Run tests
	cd backend && pytest tests/ -v --tb=short

lint: ## Run linter
	cd backend && ruff check app/ tests/

run: ## Start FastAPI server
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Start Celery worker
	cd backend && celery -A app.workers.celery_app worker --loglevel=info

docker-up: ## Start all services via Docker
	docker-compose up -d --build

docker-down: ## Stop all Docker services
	docker-compose down -v

load-data: ## Load parquet data into PostgreSQL
	cd backend && python -m scripts.load_parquet_to_db

features: ## Generate user features
	cd backend && python -m scripts.generate_features

train: ## Train the model
	cd backend && python -m scripts.train_model

migrate: ## Run database migrations
	cd backend && alembic upgrade head

migrate-create: ## Create a new migration
	cd backend && alembic revision --autogenerate -m "$(msg)"
