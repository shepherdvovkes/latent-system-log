#!/usr/bin/env python3
"""
Multithreaded Log Database Manager with connection pooling and thread-safe operations.
Optimized for high-performance read/write operations on large log datasets.
"""

import sqlite3
import os
import sys
import time
import re
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import logging
from contextlib import contextmanager
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThreadSafeConnectionPool:
    """Thread-safe connection pool for SQLite database."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._active_connections = 0
        
        # Pre-populate connection pool
        for _ in range(max_connections):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
            self._connections.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = None
        try:
            conn = self._connections.get(timeout=30)  # 30 second timeout
            yield conn
        except Empty:
            raise Exception("No available database connections")
        finally:
            if conn:
                try:
                    # Reset connection state
                    conn.rollback()
                    self._connections.put(conn)
                except:
                    # If connection is broken, create a new one
                    try:
                        new_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                        new_conn.execute("PRAGMA journal_mode=WAL")
                        new_conn.execute("PRAGMA synchronous=NORMAL")
                        new_conn.execute("PRAGMA cache_size=10000")
                        new_conn.execute("PRAGMA temp_store=MEMORY")
                        self._connections.put(new_conn)
                    except:
                        pass
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self._connections.empty():
            try:
                conn = self._connections.get_nowait()
                conn.close()
            except Empty:
                break


class MultithreadedLogDB:
    """Multithreaded log database manager with optimized operations."""
    
    def __init__(self, db_path: str = "data/multithreaded_logs.db", 
                 max_connections: int = 10, 
                 write_batch_size: int = 5000,
                 read_batch_size: int = 1000):
        self.db_path = db_path
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.connection_pool = ThreadSafeConnectionPool(db_path, max_connections)
        self.write_queue = Queue(maxsize=10000)  # Buffer for writes
        self.stats = {
            'writes': 0,
            'reads': 0,
            'errors': 0,
            'start_time': time.time()
        }
        self.stats_lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def create_database(self):
        """Create optimized database schema with proper indexing."""
        logger.info("Creating multithreaded log database...")
        
        with self.connection_pool.get_connection() as conn:
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
            
            # Create optimized indexes
            logger.info("Creating indexes for fast queries...")
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_timestamp_unix ON logs(timestamp_unix)",
                "CREATE INDEX IF NOT EXISTS idx_source ON logs(source)",
                "CREATE INDEX IF NOT EXISTS idx_level ON logs(level)",
                "CREATE INDEX IF NOT EXISTS idx_pid ON logs(pid)",
                "CREATE INDEX IF NOT EXISTS idx_created_at ON logs(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_source_level ON logs(source, level)",
                "CREATE INDEX IF NOT EXISTS idx_timestamp_source ON logs(timestamp, source)",
                "CREATE INDEX IF NOT EXISTS idx_level_timestamp ON logs(level, timestamp)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database created with optimized indexes")
    
    def _background_writer(self):
        """Background thread for batch writing to database."""
        batch = []
        
        while True:
            try:
                # Collect items from queue with timeout
                try:
                    item = self.write_queue.get(timeout=1.0)
                    batch.append(item)
                except Empty:
                    # Flush remaining batch if any
                    if batch:
                        self._flush_batch(batch)
                        batch = []
                    continue
                
                # Flush batch when it reaches the size limit
                if len(batch) >= self.write_batch_size:
                    self._flush_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Background writer error: {e}")
                if batch:
                    self._flush_batch(batch)
                    batch = []
    
    def _flush_batch(self, batch: List[tuple]):
        """Flush a batch of log entries to the database."""
        if not batch:
            return
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO logs (
                        timestamp, timestamp_unix, source, level, message,
                        thread_id, log_type, activity_id, pid, ttl, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                
                with self.stats_lock:
                    self.stats['writes'] += len(batch)
                    
        except Exception as e:
            logger.error(f"Batch flush error: {e}")
            with self.stats_lock:
                self.stats['errors'] += len(batch)
    
    def write_log_async(self, log_entry: tuple):
        """Asynchronously write a log entry to the database."""
        try:
            self.write_queue.put(log_entry, timeout=5.0)
            return True
        except Exception as e:
            logger.error(f"Failed to queue log entry: {e}")
            return False
    
    def write_logs_batch(self, log_entries: List[tuple]) -> int:
        """Write a batch of log entries synchronously."""
        if not log_entries:
            return 0
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO logs (
                        timestamp, timestamp_unix, source, level, message,
                        thread_id, log_type, activity_id, pid, ttl, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, log_entries)
                conn.commit()
                
                with self.stats_lock:
                    self.stats['writes'] += len(log_entries)
                
                return len(log_entries)
                
        except Exception as e:
            logger.error(f"Batch write error: {e}")
            with self.stats_lock:
                self.stats['errors'] += len(log_entries)
            return 0
    
    def read_logs(self, 
                  source: Optional[str] = None,
                  level: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  limit: int = 1000,
                  offset: int = 0) -> List[Dict[str, Any]]:
        """Read logs with filtering and pagination."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with conditions
                query = "SELECT * FROM logs WHERE 1=1"
                params = []
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                if level:
                    query += " AND level = ?"
                    params.append(level)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in rows:
                    result = dict(zip(columns, row))
                    if result.get('metadata'):
                        try:
                            result['metadata'] = json.loads(result['metadata'])
                        except:
                            pass
                    results.append(result)
                
                with self.stats_lock:
                    self.stats['reads'] += len(results)
                
                return results
                
        except Exception as e:
            logger.error(f"Read logs error: {e}")
            return []
    
    def read_logs_parallel(self, 
                          filters: List[Dict[str, Any]],
                          max_workers: int = 4) -> List[Dict[str, Any]]:
        """Read logs in parallel using multiple threads."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for filter_config in filters:
                future = executor.submit(
                    self.read_logs,
                    source=filter_config.get('source'),
                    level=filter_config.get('level'),
                    start_time=filter_config.get('start_time'),
                    end_time=filter_config.get('end_time'),
                    limit=filter_config.get('limit', 1000),
                    offset=filter_config.get('offset', 0)
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    logger.error(f"Parallel read error: {e}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
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
                
                with self.stats_lock:
                    stats = self.stats.copy()
                
                return {
                    "total_entries": total_count,
                    "by_level": by_level,
                    "top_sources": by_source,
                    "date_range": {"min": min_date, "max": max_date},
                    "database_size_mb": db_size,
                    "operations": stats
                }
                
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up multithreaded log database...")
        
        # Flush any remaining items in write queue
        remaining_items = []
        while not self.write_queue.empty():
            try:
                remaining_items.append(self.write_queue.get_nowait())
            except Empty:
                break
        
        if remaining_items:
            self._flush_batch(remaining_items)
        
        # Close connection pool
        self.connection_pool.close_all()


class MultithreadedLogImporter:
    """Multithreaded log importer using the thread-safe database."""
    
    def __init__(self, db_manager: MultithreadedLogDB, num_workers: int = 4):
        self.db_manager = db_manager
        self.num_workers = num_workers
        self.total_lines = 0
        self.processed_lines = 0
        self.stored_entries = 0
        self.errors = 0
        self.start_time = None
        self.lock = threading.Lock()
    
    def import_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """Import log file using multithreaded processing."""
        try:
            # Count total lines
            logger.info("Counting total lines...")
            self.total_lines = self._count_lines(log_file_path)
            logger.info(f"Found {self.total_lines:,} lines to process")
            
            # Create database
            self.db_manager.create_database()
            
            self.start_time = time.time()
            logger.info(f"Starting multithreaded import of {self.total_lines:,} lines...")
            
            # Process in chunks with multiple threads
            chunk_size = self.total_lines // self.num_workers
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for i in range(self.num_workers):
                    start_line = i * chunk_size
                    end_line = start_line + chunk_size if i < self.num_workers - 1 else self.total_lines
                    
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
            logger.info(f"Multithreaded import completed!")
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
            logger.error(f"Error during multithreaded import: {e}")
            return {"error": str(e)}
    
    def _process_chunk(self, log_file_path: str, start_line: int, end_line: int, thread_id: int) -> Dict[str, int]:
        """Process a chunk of the log file."""
        stored = 0
        errors = 0
        batch = []
        
        try:
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
                            
                            # Write batch when it reaches the size limit
                            if len(batch) >= 1000:
                                stored += self.db_manager.write_logs_batch(batch)
                                batch = []
                        
                        # Progress update every 100,000 lines
                        if (line_num - start_line) % 100000 == 0:
                            logger.info(f"Thread {thread_id}: Processed {line_num - start_line:,} lines, stored {stored:,}")
                    
                    except StopIteration:
                        break
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            logger.warning(f"Thread {thread_id} error on line {line_num}: {e}")
            
            # Write remaining batch
            if batch:
                stored += self.db_manager.write_logs_batch(batch)
            
        except Exception as e:
            logger.error(f"Thread {thread_id} failed: {e}")
            errors += 1
        
        return {"stored": stored, "errors": errors}
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[tuple]:
        """Parse a single log line."""
        try:
            pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, thread_id, log_type, activity_id, pid, ttl, rest = match.groups()
                
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f%z')
                timestamp_unix = timestamp.timestamp()
                
                if ': ' in rest:
                    source, message = rest.split(': ', 1)
                else:
                    source = rest
                    message = ""
                
                level = self._determine_log_level(message)
                
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
        """Determine log level from message."""
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
        
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count


def main():
    """Main function to demonstrate multithreaded database operations."""
    log_file = "lastday.log"
    
    if not os.path.exists(log_file):
        logger.error(f"Log file not found: {log_file}")
        return
    
    # Create multithreaded database manager
    db_manager = MultithreadedLogDB()
    
    # Create importer
    importer = MultithreadedLogImporter(db_manager, num_workers=6)
    
    # Run the import
    result = importer.import_log_file(log_file)
    
    if "error" in result:
        logger.error(f"Import failed: {result['error']}")
    else:
        logger.info("Import completed successfully!")
        
        # Get statistics
        stats = db_manager.get_statistics()
        
        print("\n" + "="*60)
        print("🚀 MULTITHREADED DATABASE IMPORT SUMMARY")
        print("="*60)
        print(f"📄 Total Lines: {result['total_lines']:,}")
        print(f"💾 Stored Entries: {result['stored_entries']:,}")
        print(f"❌ Errors: {result['errors']}")
        print(f"⏱️  Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print(f"🚀 Processing Rate: {result['processing_rate']:.0f} lines/second")
        print(f"📊 Database Size: {stats.get('database_size_mb', 0):.1f} MB")
        
        if 'operations' in stats:
            ops = stats['operations']
            print(f"📈 Operations: {ops.get('writes', 0):,} writes, {ops.get('reads', 0):,} reads, {ops.get('errors', 0)} errors")
        
        print("="*60)
        
        # Demonstrate parallel reading
        print("\n🔍 Testing parallel read operations...")
        
        # Read logs in parallel with different filters
        filters = [
            {'level': 'error', 'limit': 100},
            {'level': 'warning', 'limit': 100},
            {'source': 'kernel', 'limit': 100},
            {'source': 'bluetoothd', 'limit': 100}
        ]
        
        start_time = time.time()
        results = db_manager.read_logs_parallel(filters, max_workers=4)
        read_time = time.time() - start_time
        
        print(f"✅ Parallel read completed: {len(results)} results in {read_time:.2f} seconds")
        
        # Show sample results
        if results:
            print(f"\n📋 Sample log entry:")
            sample = results[0]
            print(f"   Timestamp: {sample.get('timestamp')}")
            print(f"   Source: {sample.get('source')}")
            print(f"   Level: {sample.get('level')}")
            print(f"   Message: {sample.get('message', '')[:100]}...")


if __name__ == "__main__":
    main()
