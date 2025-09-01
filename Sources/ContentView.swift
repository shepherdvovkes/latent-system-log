import SwiftUI

struct ContentView: View {
    @StateObject private var databaseManager = DatabaseManager()
    @StateObject private var logFetcher = LogFetcher()
    @State private var isFetching = false
    @State private var lastFetchTime: Date?
    @State private var logCount = 0
    @State private var showingAdminSettings = false
    @State private var isAdminMode = false
    @State private var showingLogViewer = false
    @State private var logs: [SystemLogEntry] = []
    @State private var selectedLogLevel: String = "ALL"
    @State private var searchText = ""
    
    private let logLevels = ["ALL", "DEBUG", "INFO", "NOTICE", "ERROR", "FAULT"]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header
                HStack {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.title2)
                        .foregroundStyle(.blue)
                    
                    Text("System Log Fetcher")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Spacer()
                    
                    Button(action: { showingAdminSettings.toggle() }) {
                        Image(systemName: "gearshape")
                            .font(.title3)
                            .foregroundStyle(isAdminMode ? .orange : .secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding()
                .background(Color(NSColor.windowBackgroundColor))
                
                Divider()
                
                // Main Content
                ScrollView {
                    VStack(spacing: 20) {
                        // Status Cards
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible())
                        ], spacing: 16) {
                            StatusCard(
                                icon: "externaldrive",
                                title: "Database",
                                value: databaseManager.isConnected ? "Connected" : "Disconnected",
                                color: databaseManager.isConnected ? .green : .red
                            )
                            
                            StatusCard(
                                icon: "doc.text",
                                title: "Logs Stored",
                                value: "\(logCount)",
                                color: .blue
                            )
                            
                            if let lastFetch = lastFetchTime {
                                StatusCard(
                                    icon: "clock",
                                    title: "Last Fetch",
                                    value: lastFetch.formatted(date: .omitted, time: .shortened),
                                    color: .orange
                                )
                            }
                            
                            StatusCard(
                                icon: "network",
                                title: "Logging Suite",
                                value: "Ready",
                                color: .purple
                            )
                        }
                        .padding(.horizontal)
                        
                        // Control Panel
                        VStack(spacing: 16) {
                            HStack {
                                Text("Log Level Filter")
                                    .font(.headline)
                                Spacer()
                                Picker("Log Level", selection: $selectedLogLevel) {
                                    ForEach(logLevels, id: \.self) { level in
                                        Text(level).tag(level)
                                    }
                                }
                                .pickerStyle(.menu)
                            }
                            
                            Button(action: fetchLogs) {
                                HStack {
                                    if isFetching {
                                        ProgressView()
                                            .scaleEffect(0.8)
                                    } else {
                                        Image(systemName: "arrow.down.circle")
                                    }
                                    Text(isFetching ? "Fetching Logs..." : "Fetch System Logs")
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(isFetching ? Color.gray.opacity(0.3) : Color.blue)
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            .disabled(isFetching)
                            
                            HStack(spacing: 12) {
                                Button(action: { showingLogViewer.toggle() }) {
                                    HStack {
                                        Image(systemName: "list.bullet")
                                        Text("View Logs")
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color(NSColor.controlBackgroundColor))
                                    .foregroundStyle(.primary)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                                
                                if isAdminMode {
                                    Button(action: clearDatabase) {
                                        HStack {
                                            Image(systemName: "trash")
                                            Text("Clear DB")
                                        }
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(Color.red)
                                        .foregroundStyle(.white)
                                        .clipShape(RoundedRectangle(cornerRadius: 10))
                                    }
                                }
                            }
                        }
                        .padding(.horizontal)
                        
                        // Recent Logs Preview
                        if !logs.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text("Recent Logs")
                                        .font(.headline)
                                    Spacer()
                                    Button("View All") {
                                        showingLogViewer.toggle()
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.blue)
                                }
                                
                                ForEach(Array(logs.prefix(3))) { log in
                                    LogEntryRow(log: log)
                                }
                            }
                            .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                }
                
                // Footer
                VStack(spacing: 4) {
                    Divider()
                    HStack {
                        Text("System Log Fetcher v2.0")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if isAdminMode {
                            Text("Admin Mode")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .fontWeight(.medium)
                        }
                    }
                    .padding(.horizontal)
                    
                    HStack {
                        Text("Vladimir Ovcharov (c) SystematicLabs 2025")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
                .background(Color(NSColor.windowBackgroundColor))
            }
        }
        .frame(minWidth: 400, minHeight: 500)
        .onAppear {
            setupDatabase()
        }
        .sheet(isPresented: $showingAdminSettings) {
            AdminSettingsView(
                databaseManager: databaseManager,
                isAdminMode: $isAdminMode
            )
        }
        .sheet(isPresented: $showingLogViewer) {
            LogViewerView(
                logs: logs,
                searchText: $searchText,
                selectedLogLevel: $selectedLogLevel
            )
        }
    }
    
    private func setupDatabase() {
        databaseManager.setupDatabase()
    }
    
    private func fetchLogs() {
        isFetching = true
        
        Task {
            do {
                let fetchedLogs = try await logFetcher.fetchSystemLogs()
                await MainActor.run {
                    self.logs = fetchedLogs
                    self.lastFetchTime = Date()
                    self.logCount += fetchedLogs.count
                    self.isFetching = false
                }
            } catch {
                await MainActor.run {
                    self.isFetching = false
                }
            }
        }
    }
    
    private func clearDatabase() {
        logCount = 0
        lastFetchTime = nil
        logs.removeAll()
    }
}

struct StatusCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(color)
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            
            Text(value)
                .font(.headline)
                .foregroundStyle(.primary)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

struct LogEntryRow: View {
    let log: SystemLogEntry
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(log.timestamp.formatted(date: .omitted, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                Spacer()
                
                Text(log.level)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(levelColor(for: log.level))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            
            Text(log.message)
                .font(.caption)
                .lineLimit(2)
                .foregroundStyle(.primary)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    
    private func levelColor(for level: String) -> Color {
        switch level.uppercased() {
        case "ERROR", "FAULT":
            return .red
        case "NOTICE":
            return .orange
        case "INFO":
            return .blue
        case "DEBUG":
            return .gray
        default:
            return .secondary
        }
    }
}

struct LogViewerView: View {
    let logs: [SystemLogEntry]
    @Binding var searchText: String
    @Binding var selectedLogLevel: String
    @Environment(\.dismiss) private var dismiss
    
    var filteredLogs: [SystemLogEntry] {
        logs.filter { log in
            let matchesLevel = selectedLogLevel == "ALL" || log.level.uppercased() == selectedLogLevel
            let matchesSearch = searchText.isEmpty || 
                log.message.localizedCaseInsensitiveContains(searchText) ||
                log.subsystem.localizedCaseInsensitiveContains(searchText)
            return matchesLevel && matchesSearch
        }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search and Filter Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    
                    TextField("Search logs...", text: $searchText)
                        .textFieldStyle(.plain)
                    
                    Picker("Level", selection: $selectedLogLevel) {
                        ForEach(["ALL", "DEBUG", "INFO", "NOTICE", "ERROR", "FAULT"], id: \.self) { level in
                            Text(level).tag(level)
                        }
                    }
                    .pickerStyle(.menu)
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                // Logs List
                List(filteredLogs) { log in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(log.timestamp.formatted())
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            
                            Spacer()
                            
                            Text(log.level)
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(levelColor(for: log.level))
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        
                        Text(log.message)
                            .font(.body)
                        
                        HStack {
                            Text("Subsystem: \(log.subsystem)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            
                            Spacer()
                            
                            Text("PID: \(log.processIdentifier)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Log Viewer")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
    
    private func levelColor(for level: String) -> Color {
        switch level.uppercased() {
        case "ERROR", "FAULT":
            return .red
        case "NOTICE":
            return .orange
        case "INFO":
            return .blue
        case "DEBUG":
            return .gray
        default:
            return .secondary
        }
    }
}

struct AdminSettingsView: View {
    @ObservedObject var databaseManager: DatabaseManager
    @Binding var isAdminMode: Bool
    @Environment(\.dismiss) private var dismiss
    @State private var adminPassword = ""
    
    var body: some View {
        VStack(spacing: 20) {
            HStack {
                Image(systemName: "lock.shield")
                    .font(.title2)
                    .foregroundStyle(.orange)
                
                Text("Admin Settings")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
            }
            
            Divider()
            
            if !isAdminMode {
                VStack(spacing: 16) {
                    Image(systemName: "lock")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    
                    Text("Enter Admin Password")
                        .font(.headline)
                    
                    SecureField("Password", text: $adminPassword)
                        .textFieldStyle(.roundedBorder)
                    
                    Button("Enable Admin Mode") {
                        if adminPassword == "admin123" {
                            isAdminMode = true
                            adminPassword = ""
                        } else {
                            adminPassword = ""
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(adminPassword.isEmpty)
                }
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Image(systemName: "externaldrive")
                            .foregroundStyle(.blue)
                        Text("Database Configuration")
                            .font(.headline)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "folder")
                                .foregroundStyle(.secondary)
                            Text("Database Path:")
                            Spacer()
                            Text(databaseManager.databasePath)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        
                        HStack {
                            Image(systemName: "checkmark.circle")
                                .foregroundStyle(databaseManager.isConnected ? .green : .red)
                            Text("Connection Status:")
                            Spacer()
                            Text(databaseManager.isConnected ? "Connected" : "Disconnected")
                                .foregroundStyle(databaseManager.isConnected ? .green : .red)
                        }
                    }
                    
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "slider.horizontal.3")
                                .foregroundStyle(.orange)
                            Text("Admin Controls")
                                .font(.headline)
                        }
                        
                        Button("Disable Admin Mode") {
                            isAdminMode = false
                        }
                        .buttonStyle(.bordered)
                        .foregroundStyle(.red)
                    }
                }
            }
            
            Spacer()
        }
        .padding()
        .frame(width: 400, height: 300)
    }
}

#Preview {
    ContentView()
}
