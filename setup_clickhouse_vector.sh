#!/bin/bash
# Setup ClickHouse + Vector for high-performance log processing

echo "🚀 Setting up ClickHouse + Vector for system log processing..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install ClickHouse
echo "📦 Installing ClickHouse..."
brew install clickhouse

# Install Vector
echo "📦 Installing Vector..."
brew install vector

# Create directories
echo "📁 Creating directories..."
mkdir -p data/clickhouse
mkdir -p config/vector
mkdir -p logs/clickhouse

# Verify installations
echo "✅ Verifying installations..."
if command -v clickhouse-server &> /dev/null; then
    echo "✅ ClickHouse installed successfully!"
    clickhouse-server --version
else
    echo "❌ Failed to install ClickHouse"
    exit 1
fi

if command -v vector &> /dev/null; then
    echo "✅ Vector installed successfully!"
    vector --version
else
    echo "❌ Failed to install Vector"
    exit 1
fi

echo "🎉 ClickHouse + Vector setup completed!"
echo ""
echo "📋 Next steps:"
echo "  1. Run: ./start_clickhouse.sh"
echo "  2. Run: ./start_vector.sh"
echo "  3. Run: ./import_logs_to_clickhouse.py"
