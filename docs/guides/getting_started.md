# Getting Started

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (recommended) or pip

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd kwikpic-backend
```

### 2. Run setup script
```bash
bash scripts/dev/setup_local.sh
```

This will:
- Copy .env.example to .env
- Install Python dependencies
- Start Docker services
- Run database migrations
- Create S3 bucket in MinIO

### 3. Start the API
```bash
# Using Poetry
poetry run uvicorn src.app.main:app --reload

# Or directly with uvicorn
uvicorn src.app.main:app --reload
```

### 4. Access the application

- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- Flower (Celery monitoring): http://localhost:5555
- MinIO Console: http://localhost:9001

## Next Steps

- Read the [API Documentation](../api/openapi.yaml)
- Check out [Development Guide](local_development.md)
- Review [Architecture Overview](../architecture/overview.md)