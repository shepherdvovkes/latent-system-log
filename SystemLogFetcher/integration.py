#!/usr/bin/env python3
"""
System Log Fetcher Integration Script
Provides integration between the Swift GUI and the logging suite
"""

import json
import sys
import os
import subprocess
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class LoggingSuiteIntegration:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config = self.load_config()
        self.db_path = self.config.get("database_path", "system_logs.db")
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "database_path": "system_logs.db",
            "log_levels": ["DEBUG", "INFO", "NOTICE", "ERROR", "FAULT"],
            "export_formats": ["json", "csv", "txt"],
            "max_logs_per_fetch": 1000,
            "python_logging_suite_path": "../app",
            "api_endpoint": "http://localhost:8000"
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return {**default_config, **json.load(f)}
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                return default_config
        else:
            # Create default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def call_logging_suite(self, command: str, **kwargs) -> Dict[str, Any]:
        """Call the Python logging suite"""
        try:
            suite_path = self.config["python_logging_suite_path"]
            
            if command == "fetch_logs":
                return self.fetch_logs_from_suite(**kwargs)
            elif command == "export_logs":
                return self.export_logs_from_suite(**kwargs)
            elif command == "analyze_logs":
                return self.analyze_logs_from_suite(**kwargs)
            elif command == "health_check":
                return self.health_check_suite()
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def fetch_logs_from_suite(self, level: str = "ALL", limit: int = 100) -> Dict[str, Any]:
        """Fetch logs from the Python logging suite"""
        try:
            # Try to call the FastAPI endpoint
            import requests
            
            url = f"{self.config['api_endpoint']}/logs"
            params = {"level": level, "limit": limit}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"error": f"API call failed: {response.status_code}"}
                
        except ImportError:
            # Fallback to direct database access
            return self.fetch_logs_from_database(level, limit)
        except Exception as e:
            return {"error": f"Failed to fetch logs: {e}"}
    
    def fetch_logs_from_database(self, level: str = "ALL", limit: int = 100) -> Dict[str, Any]:
        """Fetch logs directly from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if level == "ALL":
                query = "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?"
                cursor.execute(query, (limit,))
            else:
                query = "SELECT * FROM system_logs WHERE level = ? ORDER BY timestamp DESC LIMIT ?"
                cursor.execute(query, (level, limit))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            logs = []
            for row in rows:
                log_dict = dict(zip(columns, row))
                logs.append(log_dict)
            
            conn.close()
            return {"success": True, "data": logs, "count": len(logs)}
            
        except Exception as e:
            return {"error": f"Database error: {e}"}
    
    def export_logs_from_suite(self, format: str = "json", level: str = "ALL") -> Dict[str, Any]:
        """Export logs from the Python logging suite"""
        try:
            # Try to call the FastAPI export endpoint
            import requests
            
            url = f"{self.config['api_endpoint']}/export"
            params = {"format": format, "level": level}
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"error": f"Export API call failed: {response.status_code}"}
                
        except ImportError:
            # Fallback to local export
            return self.export_logs_locally(format, level)
        except Exception as e:
            return {"error": f"Failed to export logs: {e}"}
    
    def export_logs_locally(self, format: str = "json", level: str = "ALL") -> Dict[str, Any]:
        """Export logs locally"""
        try:
            logs_result = self.fetch_logs_from_database(level, 10000)
            if "error" in logs_result:
                return logs_result
            
            logs = logs_result["data"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_logs_{level.lower()}_{timestamp}.{format}"
            
            if format == "json":
                with open(filename, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
            elif format == "csv":
                import csv
                with open(filename, 'w', newline='') as f:
                    if logs:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        writer.writerows(logs)
            elif format == "txt":
                with open(filename, 'w') as f:
                    f.write(f"System Logs Export\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Level: {level}\n")
                    f.write(f"Total Entries: {len(logs)}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for log in logs:
                        f.write(f"[{log.get('timestamp', 'N/A')}] [{log.get('level', 'N/A')}] [{log.get('subsystem', 'N/A')}]\n")
                        f.write(f"Process: {log.get('process_name', 'N/A')} (PID: {log.get('process_identifier', 'N/A')})\n")
                        f.write(f"Message: {log.get('message', 'N/A')}\n")
                        f.write("-" * 30 + "\n")
            
            return {"success": True, "filename": filename, "count": len(logs)}
            
        except Exception as e:
            return {"error": f"Export failed: {e}"}
    
    def analyze_logs_from_suite(self) -> Dict[str, Any]:
        """Analyze logs from the Python logging suite"""
        try:
            # Try to call the FastAPI analysis endpoint
            import requests
            
            url = f"{self.config['api_endpoint']}/analyze"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"error": f"Analysis API call failed: {response.status_code}"}
                
        except ImportError:
            # Fallback to local analysis
            return self.analyze_logs_locally()
        except Exception as e:
            return {"error": f"Failed to analyze logs: {e}"}
    
    def analyze_logs_locally(self) -> Dict[str, Any]:
        """Analyze logs locally"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic statistics
            cursor.execute("SELECT COUNT(*) FROM system_logs")
            total_logs = cursor.fetchone()[0]
            
            cursor.execute("SELECT level, COUNT(*) FROM system_logs GROUP BY level")
            level_counts = dict(cursor.fetchall())
            
            cursor.execute("SELECT subsystem, COUNT(*) FROM system_logs GROUP BY subsystem ORDER BY COUNT(*) DESC LIMIT 10")
            top_subsystems = cursor.fetchall()
            
            cursor.execute("SELECT process_name, COUNT(*) FROM system_logs GROUP BY process_name ORDER BY COUNT(*) DESC LIMIT 10")
            top_processes = cursor.fetchall()
            
            conn.close()
            
            return {
                "success": True,
                "data": {
                    "total_logs": total_logs,
                    "level_distribution": level_counts,
                    "top_subsystems": top_subsystems,
                    "top_processes": top_processes,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def health_check_suite(self) -> Dict[str, Any]:
        """Check the health of the logging suite"""
        try:
            # Try to call the FastAPI health endpoint
            import requests
            
            url = f"{self.config['api_endpoint']}/health"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return {"success": True, "status": "healthy", "data": response.json()}
            else:
                return {"success": False, "status": "unhealthy", "error": f"HTTP {response.status_code}"}
                
        except ImportError:
            # Fallback to local health check
            return self.health_check_local()
        except Exception as e:
            return {"success": False, "status": "error", "error": str(e)}
    
    def health_check_local(self) -> Dict[str, Any]:
        """Check local system health"""
        try:
            # Check if database exists and is accessible
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM system_logs")
                log_count = cursor.fetchone()[0]
                conn.close()
                
                return {
                    "success": True,
                    "status": "healthy",
                    "data": {
                        "database_accessible": True,
                        "log_count": log_count,
                        "database_path": self.db_path
                    }
                }
            else:
                return {
                    "success": False,
                    "status": "unhealthy",
                    "error": "Database not found"
                }
                
        except Exception as e:
            return {"success": False, "status": "error", "error": str(e)}

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python integration.py <command> [options]")
        print("Commands: fetch_logs, export_logs, analyze_logs, health_check")
        sys.exit(1)
    
    command = sys.argv[1]
    integration = LoggingSuiteIntegration()
    
    # Parse additional arguments
    kwargs = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            kwargs[key] = value
    
    result = integration.call_logging_suite(command, **kwargs)
    
    # Output result as JSON
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
