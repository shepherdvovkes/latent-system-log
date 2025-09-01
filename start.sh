#!/bin/bash

# System Log Analysis and AI Assistant Startup Script

echo "Starting System Log Analysis and AI Assistant..."

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is not installed. Please install Python 3.12 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3.12 &> /dev/null; then
    echo "Error: pip3.12 is not installed. Please install pip3.12."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    python3.12 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip to latest version
echo "Upgrading pip to latest version..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data models logs

# Set environment variables if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp config.env.example .env
    echo "Please review and update .env file with your configuration."
fi

# Start the application
echo "Starting the application server..."
echo "The application will be available at: http://localhost:8000"
echo "API documentation will be available at: http://localhost:8000/docs"
echo "Web interface will be available at: http://localhost:8000/web_interface.html"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python main.py
