#!/bin/bash
set -e

echo "?? Setting up Kwikpic local development environment..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "?? Creating .env file..."
    cp .env.example .env
    echo "??  Please update .env with your configuration"
fi

# Install dependencies
echo "?? Installing dependencies..."
poetry install

# Start Docker services
echo "?? Starting Docker services..."
docker-compose up -d postgres redis minio

# Wait for services
echo "? Waiting for services to be ready..."
sleep 10

# Run migrations
echo "?? Running database migrations..."
poetry run alembic upgrade head

# Create MinIO bucket
echo "?? Creating S3 bucket in MinIO..."
docker-compose exec -T minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker-compose exec -T minio mc mb local/kwikpic-photos --ignore-existing

echo "? Setup complete! Run 'poetry run uvicorn src.app.main:app --reload' to start the API"
