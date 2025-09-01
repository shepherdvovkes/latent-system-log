import SwiftUI

struct ContentView: View {
    @StateObject private var databaseManager = DatabaseManager()
    @State private var isFetching = false
    @State private var lastFetchTime: Date?
    @State private var logCount = 0
    @State private var showingAdminSettings = false
    @State private var isAdminMode = false
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.title2)
                    .foregroundStyle(Color.blue)
                
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
            .padding(.horizontal)
            
            Divider()
            
            // Status Section
            VStack(spacing: 8) {
                HStack {
                    Image(systemName: "externaldrive")
                        .foregroundStyle(.green)
                    Text("Database:")
                    Text(databaseManager.isConnected ? "Connected" : "Disconnected")
                        .foregroundStyle(databaseManager.isConnected ? .green : Color.red)
                    Spacer()
                }
                
                HStack {
                    Image(systemName: "doc.text")
                        .foregroundStyle(Color.blue)
                    Text("Logs Stored:")
                    Text("\(logCount)")
                        .foregroundStyle(Color.blue)
                    Spacer()
                }
                
                if let lastFetch = lastFetchTime {
                    HStack {
                        Image(systemName: "clock")
                            .foregroundStyle(.orange)
                        Text("Last Fetch:")
                        Text(lastFetch, style: .time)
                            .foregroundStyle(.orange)
                        Spacer()
                    }
                }
            }
            .padding(.horizontal)
            
            Divider()
            
            // Main Control
            VStack(spacing: 12) {
                Button(action: fetchLogs) {
                    HStack {
                        if isFetching {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "arrow.down.circle")
                        }
                        Text(isFetching ? "Fetching..." : "Fetch System Logs")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isFetching ? Color.gray.opacity(0.3) : Color.blue)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .disabled(isFetching)
                
                if isAdminMode {
                    Button(action: clearDatabase) {
                        HStack {
                            Image(systemName: "trash")
                            Text("Clear Database")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .disabled(isFetching)
                }
            }
            .padding(.horizontal)
            
            Spacer()
            
            // Footer
            VStack(spacing: 4) {
                HStack {
                    Text("System Log Fetcher v1.0")
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
                
                HStack {
                    Text("Vladimir Ovcharov (c) SystematicLabs 2025")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            .padding(.horizontal)
        }
        .frame(width: 350, height: 400)
        .padding()
        .onAppear {
            setupDatabase()
        }
        .sheet(isPresented: $showingAdminSettings) {
            AdminSettingsView(
                databaseManager: databaseManager,
                isAdminMode: $isAdminMode
            )
        }
    }
    
    private func setupDatabase() {
        databaseManager.setupDatabase()
    }
    
    private func fetchLogs() {
        isFetching = true
        
        // Simulate log fetching
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.lastFetchTime = Date()
            self.logCount += 10
            self.isFetching = false
        }
    }
    
    private func clearDatabase() {
        logCount = 0
        lastFetchTime = nil
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
                            .foregroundStyle(Color.blue)
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
                                .foregroundStyle(databaseManager.isConnected ? .green : Color.red)
                            Text("Connection Status:")
                            Spacer()
                            Text(databaseManager.isConnected ? "Connected" : "Disconnected")
                                .foregroundStyle(databaseManager.isConnected ? .green : Color.red)
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
                        .foregroundStyle(Color.red)
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
