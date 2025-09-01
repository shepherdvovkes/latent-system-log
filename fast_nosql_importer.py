#!/usr/bin/env python3
"""
Fast NoSQL-style bulk importer for large log files.
Uses SQLite with optimized bulk inserts and indexing.
"""

import sqlite3
import os
import sys
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastNoSQLLogImporter:
    """Fast NoSQL-style importer using SQLite with bulk operations."""
    
    def __init__(self, db_path: str = "data/fast_logs.db"):
        self.db_path = db_path
        self.batch_size = 10000  # Larger batches for better performance
        self.total_lines = 0
        self.processed_lines = 0
        self.stored_entries = 0
        self.errors = 0
        self.start_time = None
        self.lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    def create_database(self):
        """Create optimized SQLite database with proper indexing."""
        logger.info("Creating optimized SQLite database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create optimized table structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                timestamp_unix REAL,
                source TEXT,
                level TEXT,
                message TEXT,
                thread_id TEXT,
                log_type TEXT,
                activity_id TEXT,
                pid INTEGER,
                ttl INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast queries
        logger.info("Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_unix ON logs(timestamp_unix)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON logs(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON logs(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pid ON logs(pid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON logs(created_at)")
        
        # Create composite indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_level ON logs(source, level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_source ON logs(timestamp, source)")
        
        conn.commit()
        conn.close()
        logger.info("Database created with optimized indexes")
    
    def import_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """Import the entire log file using fast bulk operations."""
        try:
            # Count total lines
            logger.info("Counting total lines...")
            self.total_lines = self._count_lines(log_file_path)
            logger.info(f"Found {self.total_lines:,} lines to process")
            
            # Create database
            self.create_database()
            
            self.start_time = time.time()
            logger.info(f"Starting fast bulk import of {self.total_lines:,} lines...")
            
            # Use multiple threads for processing
            num_threads = 4
            chunk_size = self.total_lines // num_threads
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                
                for i in range(num_threads):
                    start_line = i * chunk_size
                    end_line = start_line + chunk_size if i < num_threads - 1 else self.total_lines
                    
                    future = executor.submit(
                        self._process_chunk, 
                        log_file_path, 
                        start_line, 
                        end_line,
                        i
                    )
                    futures.append(future)
                
                # Wait for all threads to complete
                for future in as_completed(futures):
                    result = future.result()
                    with self.lock:
                        self.stored_entries += result['stored']
                        self.errors += result['errors']
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Fast bulk import completed!")
            logger.info(f"Total stored: {self.stored_entries:,} entries")
            logger.info(f"Total errors: {self.errors}")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Processing rate: {self.total_lines/elapsed_time:.0f} lines/second")
            
            return {
                "total_lines": self.total_lines,
                "stored_entries": self.stored_entries,
                "errors": self.errors,
                "elapsed_time": elapsed_time,
                "processing_rate": self.total_lines/elapsed_time if elapsed_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error during fast bulk import: {e}")
            return {"error": str(e)}
    
    def _process_chunk(self, log_file_path: str, start_line: int, end_line: int, thread_id: int) -> Dict[str, int]:
        """Process a chunk of the log file in a separate thread."""
        stored = 0
        errors = 0
        
        # Create thread-local database connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare bulk insert
            cursor.execute("BEGIN TRANSACTION")
            
            batch = []
            line_count = 0
            
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip to start line
                for _ in range(start_line):
                    next(f)
                
                for line_num in range(start_line, end_line):
                    try:
                        line = next(f).strip()
                        if line:
                            parsed_entry = self._parse_log_line(line, line_num)
                            if parsed_entry:
                                batch.append(parsed_entry)
                            
                            # Bulk insert when batch is full
                            if len(batch) >= self.batch_size:
                                stored += self._bulk_insert(cursor, batch)
                                batch = []
                        
                        line_count += 1
                        
                        # Progress update every 100,000 lines
                        if line_count % 100000 == 0:
                            logger.info(f"Thread {thread_id}: Processed {line_count:,} lines, stored {stored:,}")
                    
                    except StopIteration:
                        break
                    except Exception as e:
                        errors += 1
                        if errors <= 5:  # Only log first 5 errors per thread
                            logger.warning(f"Thread {thread_id} error on line {line_num}: {e}")
            
            # Insert remaining batch
            if batch:
                stored += self._bulk_insert(cursor, batch)
            
            cursor.execute("COMMIT")
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Thread {thread_id} failed: {e}")
            errors += 1
        finally:
            conn.close()
        
        return {"stored": stored, "errors": errors}
    
    def _bulk_insert(self, cursor, batch: List[tuple]) -> int:
        """Perform bulk insert of log entries."""
        try:
            cursor.executemany("""
                INSERT INTO logs (
                    timestamp, timestamp_unix, source, level, message,
                    thread_id, log_type, activity_id, pid, ttl, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            return len(batch)
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            return 0
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[tuple]:
        """Parse a single log line and return tuple for bulk insert."""
        try:
            # Parse the structured log format
            pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, thread_id, log_type, activity_id, pid, ttl, rest = match.groups()
                
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f%z')
                timestamp_unix = timestamp.timestamp()
                
                # Split source and message
                if ': ' in rest:
                    source, message = rest.split(': ', 1)
                else:
                    source = rest
                    message = ""
                
                # Determine log level
                level = self._determine_log_level(message)
                
                # Create metadata JSON
                metadata = json.dumps({
                    'thread_id': thread_id,
                    'log_type': log_type,
                    'activity_id': activity_id,
                    'ttl': int(ttl) if ttl.isdigit() else None,
                    'line_number': line_num
                })
                
                return (
                    timestamp_str,
                    timestamp_unix,
                    source,
                    level,
                    message,
                    thread_id,
                    log_type,
                    activity_id,
                    int(pid) if pid.isdigit() else None,
                    int(ttl) if ttl.isdigit() else None,
                    metadata
                )
            
            return None
            
        except Exception as e:
            return None
    
    def _determine_log_level(self, message: str) -> str:
        """Determine log level from message content."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['error', 'failed', 'failure', 'deny', 'violation', 'exception']):
            return 'error'
        elif any(word in message_lower for word in ['warning', 'warn', 'deprecated']):
            return 'warning'
        elif any(word in message_lower for word in ['debug', 'trace']):
            return 'debug'
        else:
            return 'info'
    
    def _count_lines(self, file_path: str) -> int:
        """Count total lines efficiently."""
        try:
            import subprocess
            result = subprocess.run(['wc', '-l', file_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.strip().split()[0])
        except:
            pass
        
        # Fallback
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total count
            cursor.execute("SELECT COUNT(*) FROM logs")
            total_count = cursor.fetchone()[0]
            
            # Count by level
            cursor.execute("SELECT level, COUNT(*) FROM logs GROUP BY level")
            by_level = dict(cursor.fetchall())
            
            # Count by source
            cursor.execute("SELECT source, COUNT(*) FROM logs GROUP BY source ORDER BY COUNT(*) DESC LIMIT 10")
            by_source = dict(cursor.fetchall())
            
            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM logs")
            min_date, max_date = cursor.fetchone()
            
            # Database size
            db_size = os.path.getsize(self.db_path) / (1024*1024)  # MB
            
            return {
                "total_entries": total_count,
                "by_level": by_level,
                "top_sources": by_source,
                "date_range": {"min": min_date, "max": max_date},
                "database_size_mb": db_size
            }
            
        finally:
            conn.close()


def main():
    """Main function to run the fast NoSQL import."""
    log_file = "lastday.log"
    
    if not os.path.exists(log_file):
        logger.error(f"Log file not found: {log_file}")
        return
    
    # Create fast importer
    importer = FastNoSQLLogImporter()
    
    # Run the import
    result = importer.import_log_file(log_file)
    
    if "error" in result:
        logger.error(f"Import failed: {result['error']}")
    else:
        logger.info("Import completed successfully!")
        
        # Get statistics
        stats = importer.get_statistics()
        
        print("\n" + "="*60)
        print("🚀 FAST NOSQL IMPORT SUMMARY")
        print("="*60)
        print(f"📄 Total Lines: {result['total_lines']:,}")
        print(f"💾 Stored Entries: {result['stored_entries']:,}")
        print(f"❌ Errors: {result['errors']}")
        print(f"⏱️  Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print(f"🚀 Processing Rate: {result['processing_rate']:.0f} lines/second")
        print(f"📊 Database Size: {stats['database_size_mb']:.1f} MB")
        print(f"📅 Date Range: {stats['date_range']['min']} to {stats['date_range']['max']}")
        print("\n📈 Log Levels:")
        for level, count in stats['by_level'].items():
            print(f"   {level}: {count:,}")
        print("\n🔝 Top Sources:")
        for source, count in list(stats['top_sources'].items())[:5]:
            print(f"   {source}: {count:,}")
        print("="*60)


if __name__ == "__main__":
    main()
