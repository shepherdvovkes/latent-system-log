#!/bin/bash
# Start ClickHouse + Vector Docker stack

echo "🚀 Starting ClickHouse + Vector Docker stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p config/clickhouse
mkdir -p scripts
mkdir -p logs

# Copy initialization script
if [ ! -f "scripts/init_clickhouse.sql" ]; then
    echo "❌ Missing init_clickhouse.sql script"
    exit 1
fi

# Check if lastday.log exists
if [ ! -f "lastday.log" ]; then
    echo "❌ lastday.log not found in current directory"
    exit 1
fi

echo "📦 Pulling Docker images..."
docker-compose pull

echo "🚀 Starting services..."
docker-compose up -d

# Wait for ClickHouse to be ready
echo "⏳ Waiting for ClickHouse to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:8123/?query=SELECT%201" | grep -q "1"; then
        echo "✅ ClickHouse is ready!"
        break
    fi
    echo "   Waiting... ($i/30)"
    sleep 2
done

# Initialize ClickHouse database
echo "📊 Initializing ClickHouse database..."
docker-compose exec clickhouse clickhouse-client --query "$(cat scripts/init_clickhouse.sql)"

# Check Vector status
echo "📈 Checking Vector status..."
sleep 5
if curl -s "http://localhost:8686/api/v1/health" > /dev/null; then
    echo "✅ Vector is running!"
else
    echo "⚠️  Vector may still be starting..."
fi

echo ""
echo "🎉 Docker stack is running!"
echo "📊 ClickHouse: http://localhost:8123"
echo "📈 Vector API: http://localhost:8686"
echo ""
echo "💡 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Check status: docker-compose ps"
echo ""
echo "🔍 Monitor processing:"
echo "   curl 'http://localhost:8123/?query=SELECT%20count()%20FROM%20system_logs.logs'"
echo "   docker-compose logs -f vector"
