#!/bin/bash
# Start Vector for log processing

echo "🚀 Starting Vector for log processing..."

# Check if ClickHouse is running
if ! curl -s "http://localhost:8123/?query=SELECT%201" | grep -q "1"; then
    echo "❌ ClickHouse is not running. Please start it first with: ./start_clickhouse.sh"
    exit 1
fi

# Create ClickHouse database and table
echo "📊 Setting up ClickHouse database..."

curl -s "http://localhost:8123/?query=CREATE%20DATABASE%20IF%20NOT%20EXISTS%20system_logs" > /dev/null

# Create optimized table for system logs
curl -s "http://localhost:8123/?query=CREATE%20TABLE%20IF%20NOT%20EXISTS%20system_logs.logs%20(%0A%20%20%20%20timestamp%20DateTime64(3)%2C%0A%20%20%20%20timestamp_unix%20Float64%2C%0A%20%20%20%20source%20String%2C%0A%20%20%20%20level%20Enum8('info'%20%3D%201%2C%20'warning'%20%3D%202%2C%20'error'%20%3D%203%2C%20'debug'%20%3D%204)%2C%0A%20%20%20%20message%20String%2C%0A%20%20%20%20thread_id%20String%2C%0A%20%20%20%20log_type%20String%2C%0A%20%20%20%20activity_id%20String%2C%0A%20%20%20%20pid%20UInt32%2C%0A%20%20%20%20ttl%20UInt32%2C%0A%20%20%20%20host%20String%2C%0A%20%20%20%20processor%20String%2C%0A%20%20%20%20ingestion_time%20DateTime64(3)%2C%0A%20%20%20%20metadata%20String%0A)%20ENGINE%20%3D%20MergeTree()%0A%20%20%20%20PARTITION%20BY%20toYYYYMM(timestamp)%0A%20%20%20%20ORDER%20BY%20(timestamp%2C%20source%2C%20level)%0A%20%20%20%20TTL%20timestamp%20%2B%20INTERVAL%201%20YEAR%0A%20%20%20%20SETTINGS%20index_granularity%20%3D%208192" > /dev/null

echo "✅ ClickHouse table created successfully!"

# Start Vector
echo "📈 Starting Vector with configuration..."
vector --config config/vector.toml &

# Wait for Vector to start
sleep 3

# Check if Vector is running
if pgrep -f "vector" > /dev/null; then
    echo "✅ Vector started successfully!"
    echo "🌐 Vector API: http://localhost:8686"
    echo "📊 Processing logs and sending to ClickHouse..."
    
    # Show Vector status
    echo ""
    echo "📈 Vector Status:"
    curl -s "http://localhost:8686/api/v1/health" | python3 -m json.tool 2>/dev/null || echo "API not ready yet"
    
else
    echo "❌ Failed to start Vector"
    exit 1
fi

echo ""
echo "🎉 ClickHouse + Vector setup is running!"
echo "📊 ClickHouse: http://localhost:8123"
echo "📈 Vector: http://localhost:8686"
echo ""
echo "💡 To monitor processing:"
echo "   tail -f logs/vector_errors.log"
echo "   curl 'http://localhost:8123/?query=SELECT%20count()%20FROM%20system_logs.logs'"
