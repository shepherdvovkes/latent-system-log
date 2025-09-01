import Foundation
import SQLite3

class DatabaseManager: ObservableObject {
    @Published var isConnected = false
    @Published var databasePath: String = ""
    
    private var database: OpaquePointer?
    
    init() {
        setupDatabasePath()
    }
    
    private func setupDatabasePath() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let databaseURL = documentsPath.appendingPathComponent("system_logs.db")
        databasePath = databaseURL.path
    }
    
    func setupDatabase() {
        DispatchQueue.global(qos: .background).async {
            self.createDatabase()
        }
    }
    
    private func createDatabase() {
        guard sqlite3_open(databasePath, &database) == SQLITE_OK else {
            print("Error opening database")
            return
        }
        
        let createTableSQL = """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                subsystem TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                process_name TEXT NOT NULL,
                process_identifier INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """
        
        if sqlite3_exec(database, createTableSQL, nil, nil, nil) == SQLITE_OK {
            DispatchQueue.main.async {
                self.isConnected = true
            }
            print("Database created successfully")
        } else {
            print("Error creating table")
        }
    }
    
    deinit {
        if let database = database {
            sqlite3_close(database)
        }
    }
}
