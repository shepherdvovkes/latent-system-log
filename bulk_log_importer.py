#!/usr/bin/env python3
"""
Bulk log importer for processing large log files into the database.
Optimized for handling the lastday.log file with 20M+ lines.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import re
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import AsyncSessionLocal
from app.models.database_models import LogEntryDB
from app.models.system_log_models import SystemLogEntry


class BulkLogImporter:
    """Efficient bulk importer for large log files."""
    
    def __init__(self, log_file_path: str, batch_size: int = 1000):
        self.log_file_path = log_file_path
        self.batch_size = batch_size
        self.total_lines = 0
        self.processed_lines = 0
        self.stored_entries = 0
        self.errors = 0
        self.start_time = None
        
    async def import_log_file(self) -> Dict[str, Any]:
        """Import the entire log file into the database."""
        try:
            # Count total lines first
            logger.info("Counting total lines in log file...")
            self.total_lines = await self._count_lines()
            logger.info(f"Found {self.total_lines:,} lines to process")
            
            # Check if file exists
            if not os.path.exists(self.log_file_path):
                raise FileNotFoundError(f"Log file not found: {self.log_file_path}")
            
            self.start_time = time.time()
            logger.info(f"Starting bulk import of {self.total_lines:,} lines...")
            
            # Process file in batches
            batch_count = 0
            current_batch = []
            
            with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parsed_entry = self._parse_log_line(line, line_num)
                        if parsed_entry:
                            current_batch.append(parsed_entry)
                        
                        # Store batch when it reaches the batch size
                        if len(current_batch) >= self.batch_size:
                            await self._store_batch(current_batch)
                            batch_count += 1
                            current_batch = []
                            
                            # Progress update every 10 batches
                            if batch_count % 10 == 0:
                                await self._log_progress()
                    
                    self.processed_lines = line_num
                    
                    # Progress update every 100,000 lines
                    if line_num % 100000 == 0:
                        await self._log_progress()
            
            # Store remaining batch
            if current_batch:
                await self._store_batch(current_batch)
                batch_count += 1
            
            # Final progress log
            await self._log_progress()
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Bulk import completed!")
            logger.info(f"Total processed: {self.processed_lines:,} lines")
            logger.info(f"Total stored: {self.stored_entries:,} entries")
            logger.info(f"Total errors: {self.errors}")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Processing rate: {self.processed_lines/elapsed_time:.0f} lines/second")
            
            return {
                "total_lines": self.total_lines,
                "processed_lines": self.processed_lines,
                "stored_entries": self.stored_entries,
                "errors": self.errors,
                "elapsed_time": elapsed_time,
                "processing_rate": self.processed_lines/elapsed_time if elapsed_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error during bulk import: {e}")
            return {
                "error": str(e),
                "total_lines": self.total_lines,
                "processed_lines": self.processed_lines,
                "stored_entries": self.stored_entries,
                "errors": self.errors
            }
    
    async def _count_lines(self) -> int:
        """Count total lines in the file efficiently."""
        try:
            # Use wc -l for efficiency on large files
            import subprocess
            result = subprocess.run(['wc', '-l', self.log_file_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.strip().split()[0])
        except:
            pass
        
        # Fallback to Python counting
        count = 0
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single log line from lastday.log format."""
        try:
            # Parse the structured log format
            # Format: Timestamp Thread Type Activity PID TTL Source: Message
            pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, thread_id, log_type, activity_id, pid, ttl, rest = match.groups()
                
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f%z')
                
                # Split source and message
                if ': ' in rest:
                    source, message = rest.split(': ', 1)
                else:
                    source = rest
                    message = ""
                
                # Determine log level
                level = self._determine_log_level(message)
                
                # Extract metadata
                log_metadata = {
                    'thread_id': thread_id,
                    'log_type': log_type,
                    'activity_id': activity_id,
                    'ttl': int(ttl) if ttl.isdigit() else None,
                    'line_number': line_num,
                    'raw_line': line[:500]  # Limit raw line length
                }
                
                return {
                    'timestamp': timestamp,
                    'source': source,
                    'level': level,
                    'message': message,
                    'log_metadata': log_metadata
                }
            
            return None
            
        except Exception as e:
            self.errors += 1
            if self.errors <= 10:  # Only log first 10 errors
                logger.warning(f"Failed to parse line {line_num}: {line[:100]}... Error: {e}")
            return None
    
    def _determine_log_level(self, message: str) -> str:
        """Determine log level from message content."""
        message_lower = message.lower()
        
        # Error indicators
        if any(word in message_lower for word in ['error', 'failed', 'failure', 'deny', 'violation', 'exception']):
            return 'error'
        
        # Warning indicators
        if any(word in message_lower for word in ['warning', 'warn', 'deprecated']):
            return 'warning'
        
        # Debug indicators
        if any(word in message_lower for word in ['debug', 'trace']):
            return 'debug'
        
        # Default to info
        return 'info'
    
    async def _store_batch(self, entries: List[Dict[str, Any]]):
        """Store a batch of log entries in the database."""
        async with AsyncSessionLocal() as session:
            try:
                db_entries = []
                for entry_data in entries:
                    db_entry = LogEntryDB(**entry_data)
                    db_entries.append(db_entry)
                
                session.add_all(db_entries)
                await session.commit()
                
                self.stored_entries += len(db_entries)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing batch: {e}")
                self.errors += len(entries)
    
    async def _log_progress(self):
        """Log current progress."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            progress = (self.processed_lines / self.total_lines * 100) if self.total_lines > 0 else 0
            rate = self.processed_lines / elapsed if elapsed > 0 else 0
            
            logger.info(f"Progress: {progress:.1f}% ({self.processed_lines:,}/{self.total_lines:,}) "
                       f"| Stored: {self.stored_entries:,} | Errors: {self.errors} | "
                       f"Rate: {rate:.0f} lines/sec | Elapsed: {elapsed:.1f}s")


async def main():
    """Main function to run the bulk import."""
    log_file = "lastday.log"
    
    if not os.path.exists(log_file):
        logger.error(f"Log file not found: {log_file}")
        return
    
    # Create importer with optimized batch size
    importer = BulkLogImporter(log_file, batch_size=2000)
    
    # Run the import
    result = await importer.import_log_file()
    
    if "error" in result:
        logger.error(f"Import failed: {result['error']}")
    else:
        logger.info("Import completed successfully!")
        print("\n" + "="*60)
        print("📊 BULK IMPORT SUMMARY")
        print("="*60)
        print(f"📄 Total Lines: {result['total_lines']:,}")
        print(f"✅ Processed: {result['processed_lines']:,}")
        print(f"💾 Stored Entries: {result['stored_entries']:,}")
        print(f"❌ Errors: {result['errors']}")
        print(f"⏱️  Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print(f"🚀 Processing Rate: {result['processing_rate']:.0f} lines/second")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
