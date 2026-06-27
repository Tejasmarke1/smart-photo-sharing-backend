# smart-photo-sharing-backend

cat > README.md << 'EOF'
# Kwiksnap Backend

AI-powered event photo-sharing platform backend.

## Quick Start
```bash
# Install dependencies
poetry install

# Copy environment variables
cp .env.example .env

# Start services
docker-compose up -d

# Run migrations
alembic upgrade head

# Start API
poetry run uvicorn src.app.main:app --reload
```

## Documentation

- [Architecture](docs/architecture/overview.md)
- [API Docs](http://localhost:8000/docs)
- [Development Guide](docs/guides/local_development.md)
EOF