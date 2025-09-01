# System Log Fetcher - Swift GUI Application

A professional SwiftUI-based graphical user interface for system log collection and analysis, designed to integrate with the Python logging suite.

## Features

### 🖥️ Modern SwiftUI Interface
- **Professional Design**: Clean, modern interface following Apple Design Guidelines
- **Status Dashboard**: Real-time monitoring of database connection, log counts, and system status
- **Log Viewer**: Advanced log viewing with search and filtering capabilities
- **Admin Panel**: Secure administrative controls for database management
- **Export Functionality**: Export logs in JSON, CSV, and text formats

### 🔧 Integration Capabilities
- **Python Logging Suite**: Seamless integration with the existing Python backend
- **FastAPI Communication**: RESTful API calls to the logging suite
- **Database Access**: Direct SQLite database access for local operations
- **Real-time Updates**: Live log fetching and status updates

### 📊 Log Management
- **Multi-level Filtering**: Filter by log levels (DEBUG, INFO, NOTICE, ERROR, FAULT)
- **Search Functionality**: Full-text search across log messages and subsystems
- **Process Tracking**: Monitor specific processes and their log entries
- **Timestamp Management**: Precise timestamp tracking and formatting

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Swift GUI     │    │  Integration     │    │  Python Logging │
│   (SwiftUI)     │◄──►│  Script (Python) │◄──►│  Suite (FastAPI)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local SQLite  │    │   Configuration  │    │   System Logs   │
│   Database      │    │   Management     │    │   Collection    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation & Setup

### Prerequisites
- macOS 14.0 or later
- Xcode Command Line Tools
- Python 3.8+ (for integration)
- FastAPI/Uvicorn (for Python suite)

### Quick Start

1. **Build the Swift Application**:
   ```bash
   cd SystemLogFetcher
   ./package.sh
   ```

2. **Launch the Complete System**:
   ```bash
   ./launcher.sh start
   ```

3. **Check System Status**:
   ```bash
   ./launcher.sh status
   ```

### Manual Installation

1. **Build the App Bundle**:
   ```bash
   ./package.sh
   ```

2. **Install the Application**:
   - Drag `build/SystemLogFetcher.app` to your Applications folder
   - Or run directly: `open build/SystemLogFetcher.app`

3. **Start Python Logging Suite** (if needed):
   ```bash
   cd ../app
   source venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Swift GUI Interface

#### Main Dashboard
- **Database Status**: Shows connection status and log count
- **Logging Suite Status**: Displays integration status with Python backend
- **Fetch Controls**: One-click log collection with progress indicators
- **Recent Logs**: Quick preview of latest log entries

#### Log Viewer
- **Search**: Full-text search across all log fields
- **Level Filtering**: Filter by log severity levels
- **Sorting**: Sort by timestamp, level, or process
- **Export**: Export filtered results in multiple formats

#### Admin Panel
- **Database Management**: Clear database, view connection details
- **Configuration**: Modify integration settings
- **Security**: Password-protected administrative access

### Command Line Integration

#### Launcher Script Commands
```bash
./launcher.sh start      # Start both Swift GUI and Python suite
./launcher.sh swift      # Start only Swift GUI
./launcher.sh python     # Start only Python suite
./launcher.sh test       # Run integration tests
./launcher.sh status     # Show system status
./launcher.sh stop       # Stop all services
./launcher.sh help       # Show help
```

#### Integration Script Commands
```bash
python3 integration.py health_check
python3 integration.py fetch_logs level=INFO limit=10
python3 integration.py export_logs format=json level=ERROR
python3 integration.py analyze_logs
```

## Configuration

### Swift App Configuration
The Swift application uses system defaults but can be configured through:

- **Database Path**: Automatically set to user's Documents directory
- **Log Levels**: Configurable through the GUI
- **Export Formats**: JSON, CSV, and text formats supported

### Integration Configuration
Edit `config.json` to customize integration settings:

```json
{
  "database_path": "system_logs.db",
  "log_levels": ["DEBUG", "INFO", "NOTICE", "ERROR", "FAULT"],
  "export_formats": ["json", "csv", "txt"],
  "max_logs_per_fetch": 1000,
  "python_logging_suite_path": "../app",
  "api_endpoint": "http://localhost:8000"
}
```

## Development

### Project Structure
```
SystemLogFetcher/
├── Sources/
│   ├── SystemLogFetcherApp.swift    # Main app entry point
│   ├── ContentView.swift            # Main GUI interface
│   ├── LogFetcher.swift             # Log collection logic
│   └── DatabaseManager.swift        # Database operations
├── build/                           # Build output directory
├── package.sh                       # App packaging script
├── launcher.sh                      # System launcher
├── integration.py                   # Python integration script
└── README.md                        # This file
```

### Building from Source
```bash
# Build executable
swiftc -target x86_64-apple-macosx14.0 \
       -sdk $(xcrun --show-sdk-path) \
       -framework Foundation -framework SwiftUI -framework AppKit -framework OSLog \
       Sources/*.swift -o build/SystemLogFetcher

# Package as app bundle
./package.sh
```

### Customization
- **UI Themes**: Modify colors and styling in `ContentView.swift`
- **Log Processing**: Extend `LogFetcher.swift` for custom log handling
- **Database Schema**: Update `DatabaseManager.swift` for different data structures
- **Integration**: Modify `integration.py` for custom backend communication

## Troubleshooting

### Common Issues

#### Swift App Won't Launch
- Check macOS version compatibility (requires 14.0+)
- Verify Xcode Command Line Tools installation
- Check app bundle permissions

#### Integration Failures
- Ensure Python logging suite is running on port 8000
- Check network connectivity and firewall settings
- Verify database file permissions

#### Database Errors
- Check SQLite database file existence
- Verify write permissions in Documents directory
- Initialize database through admin panel

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export DEBUG=1
./launcher.sh start
```

## Security Considerations

- **Admin Access**: Password-protected administrative functions
- **Database Security**: Local SQLite database with user permissions
- **Network Security**: HTTPS recommended for production API endpoints
- **Log Privacy**: Logs contain system information - handle with care

## Performance

### Optimization Tips
- **Log Limits**: Set appropriate fetch limits to prevent memory issues
- **Database Indexing**: Large log databases benefit from proper indexing
- **Background Processing**: Heavy operations run in background threads
- **Caching**: Recent logs are cached for faster access

### System Requirements
- **Minimum**: macOS 14.0, 4GB RAM, 1GB disk space
- **Recommended**: macOS 14.0+, 8GB RAM, 5GB disk space
- **Network**: Local network access for API integration

## License

Copyright (c) 2025 Vladimir Ovcharov - SystematicLabs

This project is proprietary software. All rights reserved.

## Support

For technical support or feature requests:
- Check the troubleshooting section above
- Review integration logs in the admin panel
- Contact development team for advanced issues

---

**Version**: 2.0.0  
**Last Updated**: January 2025  
**Compatibility**: macOS 14.0+
