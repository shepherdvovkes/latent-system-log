#!/bin/bash
# Start ClickHouse server

echo "🚀 Starting ClickHouse server..."

# Create ClickHouse config
cat > config/clickhouse-server.xml << 'EOF'
<?xml version="1.0"?>
<clickhouse>
    <logger>
        <level>warning</level>
        <log>/Users/vovkes/loging/logs/clickhouse/clickhouse-server.log</log>
        <errorlog>/Users/vovkes/loging/logs/clickhouse/clickhouse-server.err.log</errorlog>
        <size>1000M</size>
        <count>10</count>
    </logger>

    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>

    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    <mark_cache_size>5368709120</mark_cache_size>

    <path>/Users/vovkes/loging/data/clickhouse/</path>
    <tmp_path>/Users/vovkes/loging/data/clickhouse/tmp/</tmp_path>

    <user_directories>
        <users_xml>
            <path>/Users/vovkes/loging/config/users.xml</path>
        </users_xml>
    </user_directories>

    <default_profile>default</default_profile>
    <default_database>default</default_database>

    <timezone>UTC</timezone>

    <remote_servers>
    </remote_servers>

    <include_from>/Users/vovkes/loging/config/clickhouse-server.xml</include_from>
</clickhouse>
EOF

# Create users config
cat > config/users.xml << 'EOF'
<?xml version="1.0"?>
<clickhouse>
    <users>
        <default>
            <password></password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
        </default>
    </users>

    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <use_uncompressed_cache>0</use_uncompressed_cache>
            <load_balancing>random</load_balancing>
        </default>
    </profiles>

    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
EOF

# Start ClickHouse server
echo "📊 Starting ClickHouse server on port 8123..."
clickhouse-server --config-file=config/clickhouse-server.xml &

# Wait for server to start
sleep 5

# Test connection
if curl -s "http://localhost:8123/?query=SELECT%201" | grep -q "1"; then
    echo "✅ ClickHouse server started successfully!"
    echo "🌐 Web interface: http://localhost:8123"
    echo "🔌 TCP port: 9000"
else
    echo "❌ Failed to start ClickHouse server"
    exit 1
fi
