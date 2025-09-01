#!/bin/bash

# System Log Fetcher Launcher
# This script launches both the Swift GUI and the Python logging suite

echo "System Log Fetcher Launcher"
echo "=========================="

# Configuration
SWIFT_APP_PATH="./build/SystemLogFetcher.app"
PYTHON_SUITE_PATH="../app"
INTEGRATION_SCRIPT="./integration.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to start Python logging suite
start_python_suite() {
    echo -e "${BLUE}Starting Python logging suite...${NC}"
    
    if [ ! -d "$PYTHON_SUITE_PATH" ]; then
        echo -e "${RED}Error: Python suite directory not found at $PYTHON_SUITE_PATH${NC}"
        return 1
    fi
    
    cd "$PYTHON_SUITE_PATH"
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source venv/bin/activate
    fi
    
    # Check if FastAPI server is already running
    if port_in_use 8000; then
        echo -e "${YELLOW}FastAPI server already running on port 8000${NC}"
    else
        echo -e "${GREEN}Starting FastAPI server...${NC}"
        if command_exists uvicorn; then
            uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
            PYTHON_PID=$!
            echo -e "${GREEN}Python suite started with PID: $PYTHON_PID${NC}"
        else
            echo -e "${RED}Error: uvicorn not found. Please install it with: pip install uvicorn${NC}"
            return 1
        fi
    fi
    
    cd - > /dev/null
}

# Function to start Swift app
start_swift_app() {
    echo -e "${BLUE}Starting Swift GUI...${NC}"
    
    if [ ! -d "$SWIFT_APP_PATH" ]; then
        echo -e "${RED}Error: Swift app not found at $SWIFT_APP_PATH${NC}"
        echo -e "${YELLOW}Building Swift app...${NC}"
        ./package.sh
    fi
    
    echo -e "${GREEN}Launching Swift GUI...${NC}"
    open "$SWIFT_APP_PATH"
}

# Function to run integration test
test_integration() {
    echo -e "${BLUE}Testing integration...${NC}"
    
    if [ -f "$INTEGRATION_SCRIPT" ]; then
        echo -e "${YELLOW}Running health check...${NC}"
        python3 "$INTEGRATION_SCRIPT" health_check
        
        echo -e "${YELLOW}Testing log fetch...${NC}"
        python3 "$INTEGRATION_SCRIPT" fetch_logs level=INFO limit=5
    else
        echo -e "${RED}Integration script not found${NC}"
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}System Status:${NC}"
    
    # Check Swift app
    if [ -d "$SWIFT_APP_PATH" ]; then
        echo -e "${GREEN}✓ Swift app ready${NC}"
    else
        echo -e "${RED}✗ Swift app not found${NC}"
    fi
    
    # Check Python suite
    if [ -d "$PYTHON_SUITE_PATH" ]; then
        echo -e "${GREEN}✓ Python suite directory found${NC}"
    else
        echo -e "${RED}✗ Python suite directory not found${NC}"
    fi
    
    # Check if FastAPI is running
    if port_in_use 8000; then
        echo -e "${GREEN}✓ FastAPI server running on port 8000${NC}"
    else
        echo -e "${YELLOW}⚠ FastAPI server not running${NC}"
    fi
    
    # Check integration script
    if [ -f "$INTEGRATION_SCRIPT" ]; then
        echo -e "${GREEN}✓ Integration script ready${NC}"
    else
        echo -e "${RED}✗ Integration script not found${NC}"
    fi
}

# Function to stop services
stop_services() {
    echo -e "${BLUE}Stopping services...${NC}"
    
    # Stop FastAPI server
    if port_in_use 8000; then
        echo -e "${YELLOW}Stopping FastAPI server...${NC}"
        lsof -ti:8000 | xargs kill -9
    fi
    
    # Stop Swift app
    echo -e "${YELLOW}Stopping Swift app...${NC}"
    pkill -f "SystemLogFetcher"
    
    echo -e "${GREEN}Services stopped${NC}"
}

# Function to show help
show_help() {
    echo "System Log Fetcher Launcher"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start both Swift GUI and Python suite"
    echo "  swift       Start only Swift GUI"
    echo "  python      Start only Python suite"
    echo "  test        Run integration tests"
    echo "  status      Show system status"
    echo "  stop        Stop all services"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start everything"
    echo "  $0 status   # Check what's running"
    echo "  $0 test     # Test integration"
}

# Main script logic
case "${1:-start}" in
    "start")
        echo -e "${GREEN}Starting System Log Fetcher...${NC}"
        start_python_suite
        sleep 2
        start_swift_app
        echo -e "${GREEN}System Log Fetcher started successfully!${NC}"
        echo -e "${BLUE}Swift GUI: $SWIFT_APP_PATH${NC}"
        echo -e "${BLUE}Python API: http://localhost:8000${NC}"
        ;;
    "swift")
        start_swift_app
        ;;
    "python")
        start_python_suite
        ;;
    "test")
        test_integration
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Launcher completed.${NC}"
