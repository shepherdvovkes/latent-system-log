#!/bin/bash
# Install GoAccess for fast log analysis

echo "🚀 Installing GoAccess for fast log analysis..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install GoAccess
echo "📦 Installing GoAccess..."
brew install goaccess

# Verify installation
if command -v goaccess &> /dev/null; then
    echo "✅ GoAccess installed successfully!"
    goaccess --version
else
    echo "❌ Failed to install GoAccess"
    exit 1
fi

echo "🎉 GoAccess is ready for fast log analysis!"
echo ""
echo "📋 Usage examples:"
echo "  goaccess lastday.log -o report.html --log-format=COMBINED"
echo "  goaccess lastday.log -o report.json --log-format=COMBINED --output-format=json"
echo "  goaccess lastday.log --real-time-html --port=7890"
