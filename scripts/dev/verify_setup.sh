#!/bin/bash

echo "🔍 Verifying Kwikpic setup..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if (( $(echo "$python_version >= 3.11" | bc -l) )); then
    echo "✅ Python $python_version installed"
else
    echo "❌ Python 3.11+ required (found $python_version)"
    exit 1
fi

# Check Poetry
if command -v poetry &> /dev/null; then
    echo "✅ Poetry installed"
else
    echo "❌ Poetry not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker installed"
else
    echo "❌ Docker not found"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose installed"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Docker services running"
else
    echo "⚠️  Docker services not running. Run 'docker-compose up -d'"
fi

# Check database connectivity
if docker-compose exec -T postgres pg_isready -U kwikpic &> /dev/null; then
    echo "✅ Database connection successful"
else
    echo "⚠️  Cannot connect to database"
fi

# Check Redis connectivity
if docker-compose exec -T redis redis-cli ping &> /dev/null; then
    echo "✅ Redis connection successful"
else
    echo "⚠️  Cannot connect to Redis"
fi

echo ""
echo "📋 Setup Summary:"
echo "- API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Flower: http://localhost:5555"
echo "- MinIO: http://localhost:9001"
echo ""
echo "✅ Verification complete!"
