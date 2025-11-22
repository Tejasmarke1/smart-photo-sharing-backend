# Local Development Guide

## Development Workflow

### Running the Application
```bash
# Terminal 1: Start Docker services
docker-compose up -d

# Terminal 2: Start API server
poetry run uvicorn src.app.main:app --reload

# Terminal 3: Start Celery worker
poetry run celery -A src.tasks.celery_app worker --loglevel=info

# Terminal 4: Start Celery Beat (scheduled tasks)
poetry run celery -A src.tasks.celery_app beat --loglevel=info
```

### Database Operations
```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1

# Reset database
bash scripts/dev/reset_db.sh
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/integration/api/test_auth.py

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration
```

### Code Quality
```bash
# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type checking
poetry run mypy src

# Run all checks (pre-commit)
poetry run pre-commit run --all-files
```

### Debugging
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Use Python debugger
import pdb; pdb.set_trace()
```

## Common Tasks

### Adding a New Endpoint

1. Define schema in `src/schemas/`
2. Create endpoint in `src/api/v1/endpoints/`
3. Add route to `src/api/v1/router.py`
4. Write tests in `tests/integration/api/`

### Adding a New Model

1. Create model in `src/models/`
2. Import in `src/models/__init__.py`
3. Create migration: `alembic revision --autogenerate`
4. Apply migration: `alembic upgrade head`

### Adding a Background Task

1. Create task in `src/tasks/workers/`
2. Register task in `src/tasks/celery_app.py`
3. Call task: `task_name.delay(args)`

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Redis Connection Issues
```bash
# Check if Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping
```

### Celery Worker Issues
```bash
# View worker logs
docker-compose logs celery_worker

# Inspect active tasks
poetry run celery -A src.tasks.celery_app inspect active

# Purge all tasks
poetry run celery -A src.tasks.celery_app purge