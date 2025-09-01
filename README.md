# System Log Fetcher

A macOS SwiftUI application for fetching and storing system logs with admin settings.

## Features

- **Modern SwiftUI Interface**: Clean, professional interface following Apple Design Guidelines
- **System Log Fetching**: Simulated log fetching functionality (ready for real implementation)
- **SQLite Database**: Local storage for system logs
- **Admin Settings**: Password-protected admin mode with database configuration
- **Apple Design Guidelines**: Uses SF Symbols and proper macOS design patterns

## Requirements

- macOS 15.0 (Sequoia) or later
- Swift 6.0
- Xcode 15.0 or later

## Installation

1. Clone or download the project
2. Open Terminal and navigate to the project directory
3. Build the application:
   ```bash
   swift build -Xswiftc -target -Xswiftc arm64-apple-macosx15.0
   ```
4. Run the application:
   ```bash
   .build/debug/SystemLogFetcher
   ```

## Usage

### Main Interface
- **Fetch System Logs**: Click the "Fetch System Logs" button to simulate log fetching
- **Database Status**: View database connection status and log count
- **Admin Settings**: Click the gear icon to access admin settings

### Admin Settings
- **Password**: Use "admin123" to enable admin mode
- **Database Configuration**: View database path and connection status
- **Clear Database**: Available only in admin mode

## Technical Details

### Architecture
- **SwiftUI**: Modern declarative UI framework
- **SQLite3**: Local database for log storage
- **ObservableObject**: Reactive data management
- **MainActor**: Proper concurrency handling

### File Structure
```
Sources/
├── SystemLogFetcherApp.swift    # Main app entry point
├── ContentView.swift            # Main UI and admin settings
└── DatabaseManager.swift        # Database operations
```

### Design Guidelines
- **SF Symbols**: Apple's system icons throughout the interface
- **Color Scheme**: Professional colors following macOS design
- **Typography**: System fonts with appropriate weights
- **Layout**: Responsive design with proper spacing

## Development

### Building
```bash
swift build -Xswiftc -target -Xswiftc arm64-apple-macosx15.0
```

### Running
```bash
.build/debug/SystemLogFetcher
```

### Package Management
The project uses Swift Package Manager (SPM) for dependency management.

## Copyright

Vladimir Ovcharov (c) SystematicLabs 2025

## License

This project is proprietary software. All rights reserved.
