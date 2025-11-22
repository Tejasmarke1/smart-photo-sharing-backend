.PHONY: help install setup start stop restart clean test lint format migrate shell logs

help:
    @echo "Kwikpic Backend - Available commands:"
    @echo "  make install    - Install dependencies"
    @echo "  make setup      - Setup local development environment"
    @echo "  make start      - Start all services"
    @echo "  make stop       - Stop all services"
    @echo "  make restart    - Restart all services"
    @echo "  make clean      - Clean up containers and volumes"
    @echo "  make test       - Run tests"
    @echo "  make lint       - Run linters"
    @echo "  make format     - Format code"
    @echo "  make migrate    - Run database migrations"
    @echo "  make shell      - Open Python shell"
    @echo "  make logs       - View logs"

install:
    poetry install

setup:
    bash scripts/dev/setup_local.sh

start:
    docker-compose up -d
    poetry run uvicorn src.app.main:app --reload

stop:
    docker-compose down

restart: stop start

clean:
    docker-compose down -v
    find . -type d -name __pycache__ -exec rm -r {} +
    find . -type f -name '*.pyc' -delete
    rm -rf .pytest_cache .coverage htmlcov

test:
    poetry run pytest

lint:
    poetry run black --check src tests
    poetry run ruff check src tests
    poetry run mypy src

format:
    poetry run black src tests
    poetry run ruff check --fix src tests

migrate:
    poetry run alembic upgrade head

shell:
    poetry run python

logs:
    docker-compose logs -f
