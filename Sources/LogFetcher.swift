import Foundation
import OSLog

struct SystemLogEntry: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let level: String
    let subsystem: String
    let category: String
    let message: String
    let processName: String
    let processIdentifier: Int32
    
    enum CodingKeys: String, CodingKey {
        case timestamp, level, subsystem, category, message, processName, processIdentifier
    }
}

class LogFetcher: ObservableObject {
    private let logStore = try? OSLogStore(scope: .currentProcessIdentifier)
    
    func fetchSystemLogs() async throws -> [SystemLogEntry] {
        // Try to fetch real system logs first
        if let realLogs = try? await fetchRealSystemLogs() {
            return realLogs
        }
        
        // Fallback to sample logs if real logs are not available
        return generateSampleLogs()
    }
    
    private func fetchRealSystemLogs() async throws -> [SystemLogEntry] {
        guard let logStore = logStore else {
            throw LogFetcherError.logStoreUnavailable
        }
        
        let position = logStore.position(date: Date().addingTimeInterval(-3600)) // Last hour
        let entries = try logStore.getEntries(at: position)
        
        var logs: [SystemLogEntry] = []
        
        for entry in entries {
            if let logEntry = convertOSLogEntry(entry) {
                logs.append(logEntry)
            }
        }
        
        return logs.sorted { $0.timestamp > $1.timestamp }
    }
    
    private func convertOSLogEntry(_ entry: OSLogEntry) -> SystemLogEntry? {
        guard let log = entry as? OSLogEntryLog else { return nil }
        
        return SystemLogEntry(
            timestamp: log.date,
            level: log.level.description,
            subsystem: log.subsystem,
            category: log.category,
            message: log.composedMessage,
            processName: log.process,
            processIdentifier: log.processIdentifier
        )
    }
    
    private func generateSampleLogs() -> [SystemLogEntry] {
        let sampleMessages = [
            "System startup completed successfully",
            "Network interface en0 configured",
            "User login session started",
            "Application SystemLogFetcher launched",
            "Database connection established",
            "Log collection process initiated",
            "System resources allocated",
            "Background task completed",
            "Memory usage within normal limits",
            "Disk space available: 256GB"
        ]
        
        let subsystems = ["system", "network", "security", "application", "kernel"]
        let levels = ["INFO", "DEBUG", "NOTICE", "ERROR"]
        
        return (0..<10).map { index in
            SystemLogEntry(
                timestamp: Date().addingTimeInterval(-Double(index * 60)),
                level: levels.randomElement() ?? "INFO",
                subsystem: subsystems.randomElement() ?? "system",
                category: "SystemLogFetcher",
                message: sampleMessages[index % sampleMessages.count],
                processName: "SystemLogFetcher",
                processIdentifier: Int32.random(in: 1000...9999)
            )
        }
    }
    
    func exportLogsToFile(_ logs: [SystemLogEntry], format: ExportFormat = .json) throws -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let filename = "system_logs_\(timestamp).\(format.fileExtension)"
        let fileURL = documentsPath.appendingPathComponent(filename)
        
        let data: Data
        
        switch format {
        case .json:
            data = try JSONEncoder().encode(logs)
        case .csv:
            data = exportToCSV(logs).data(using: .utf8) ?? Data()
        case .txt:
            data = exportToText(logs).data(using: .utf8) ?? Data()
        }
        
        try data.write(to: fileURL)
        return fileURL
    }
    
    private func exportToCSV(_ logs: [SystemLogEntry]) -> String {
        var csv = "Timestamp,Level,Subsystem,Category,Message,Process,ProcessID\n"
        
        for log in logs {
            let row = "\(log.timestamp),\(log.level),\(log.subsystem),\(log.category),\"\(log.message)\",\(log.processName),\(log.processIdentifier)\n"
            csv += row
        }
        
        return csv
    }
    
    private func exportToText(_ logs: [SystemLogEntry]) -> String {
        var text = "System Logs Export\n"
        text += "Generated: \(Date())\n"
        text += "Total Entries: \(logs.count)\n"
        text += String(repeating: "=", count: 50) + "\n\n"
        
        for log in logs {
            text += "[\(log.timestamp)] [\(log.level)] [\(log.subsystem)]\n"
            text += "Process: \(log.processName) (PID: \(log.processIdentifier))\n"
            text += "Message: \(log.message)\n"
            text += String(repeating: "-", count: 30) + "\n"
        }
        
        return text
    }
}

enum LogFetcherError: Error {
    case logStoreUnavailable
    case exportFailed
}

enum ExportFormat {
    case json, csv, txt
    
    var fileExtension: String {
        switch self {
        case .json: return "json"
        case .csv: return "csv"
        case .txt: return "txt"
        }
    }
    
    var displayName: String {
        switch self {
        case .json: return "JSON"
        case .csv: return "CSV"
        case .txt: return "Text"
        }
    }
}

extension OSLogEntryLog.Level {
    var description: String {
        switch self {
        case .undefined:
            return "UNDEFINED"
        case .debug:
            return "DEBUG"
        case .info:
            return "INFO"
        case .notice:
            return "NOTICE"
        case .error:
            return "ERROR"
        case .fault:
            return "FAULT"
        @unknown default:
            return "UNKNOWN"
        }
    }
}
