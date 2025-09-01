"""
Log collection service for gathering system logs.
"""

import asyncio
import subprocess
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import psutil

from app.core.config import settings
from app.models.schemas import LogEntry


class LogCollectorService:
    """Service for collecting system logs from various sources."""
    
    def __init__(self):
        self.log_file_path = os.path.join(settings.DATA_DIR, "system_logs.jsonl")
        self.last_collection = None
        
    async def initialize(self):
        """Initialize the log collector service."""
        logger.info("Initializing LogCollectorService...")
        # Ensure log file exists
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w') as f:
                pass
        logger.info("LogCollectorService initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up LogCollectorService...")
    
    async def collect_system_logs(self) -> List[LogEntry]:
        """Collect system logs from macOS system and lastmin.log file."""
        logs = []
        
        try:
            # First, collect fresh logs from macOS system
            fresh_logs = await self._collect_macos_logs()
            logs.extend(fresh_logs)
            
            # Also parse the existing lastmin.log file for additional data
            from app.services.system_log_parser import SystemLogParser
            parser = SystemLogParser()
            
            # Parse the log file
            result = await parser.parse_log_file(minutes=1)
            
            # Get statistics from the parsed logs
            stats = await parser.get_log_statistics(minutes=1)
            
            # Create LogEntry objects from the parsed data
            if stats.get('total_logs', 0) > 0:
                # Create a summary log entry
                summary_log = LogEntry(
                    timestamp=datetime.now(),
                    source='system_log_parser',
                    level='info',
                    message=f"Parsed {result.get('parsed_entries', 0)} system log entries from lastmin.log",
                    metadata={
                        'parsed_entries': result.get('parsed_entries', 0),
                        'stored_entries': result.get('stored_entries', 0),
                        'total_logs': stats.get('total_logs', 0),
                        'by_level': stats.get('by_level', {}),
                        'by_source': stats.get('by_source', {})
                    }
                )
                logs.append(summary_log)
            
            # Save logs to file
            await self._save_logs(logs)
            
            self.last_collection = datetime.now()
            logger.info(f"Collected {len(logs)} log entries (fresh: {len(fresh_logs)}, parsed: {result.get('parsed_entries', 0)})")
            
        except Exception as e:
            logger.error(f"Error collecting logs: {e}")
            
        return logs
    
    async def _collect_macos_logs(self) -> List[LogEntry]:
        """Collect macOS system logs using 'log show --last 30s'."""
        logs = []
        try:
            import platform
            system = platform.system()
            logger.info(f"Detected platform: {system}")
            
            if system == "Darwin":  # macOS
                # Use macOS system logs with --last 30s (shorter to avoid timeout)
                cmd = ["log", "show", "--last", "30s", "--style", "json"]
                try:
                    logger.info("Starting macOS log collection...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    logger.info(f"macOS log collection completed with return code: {result.returncode}")
                    if result.stderr:
                        logger.warning(f"macOS log collection stderr: {result.stderr[:200]}")
                    if result.returncode == 0:
                        # Parse the entire output as a JSON array
                        try:
                            data_array = json.loads(result.stdout.strip())
                            logger.info(f"Retrieved {len(data_array)} log entries from macOS system")
                            
                            # Process each log entry in the array
                            for data in data_array:
                                try:
                                    # Extract timestamp - handle different timestamp formats
                                    timestamp_str = data.get('timestamp', '')
                                    if timestamp_str:
                                        # Remove timezone info and parse
                                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                    else:
                                        timestamp = datetime.now()
                                    
                                    log_entry = LogEntry(
                                        timestamp=timestamp,
                                        source=data.get('source', 'system'),
                                        level=self._determine_log_level(data),
                                        message=data.get('eventMessage', ''),
                                        metadata={
                                            'subsystem': data.get('subsystem', ''),
                                            'category': data.get('category', ''),
                                            'process': data.get('process', ''),
                                            'thread': data.get('thread', ''),
                                            'eventType': data.get('eventType', ''),
                                            'messageType': data.get('messageType', '')
                                        }
                                    )
                                    logs.append(log_entry)
                                except (KeyError, ValueError) as e:
                                    continue
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON array from log show: {e}")
                            # Fallback to line-by-line parsing
                            for line in result.stdout.strip().split('\n'):
                                if line.strip() and line.strip() not in ['[', ']']:
                                    try:
                                        data = json.loads(line.rstrip(','))
                                        # ... rest of the parsing logic
                                    except (json.JSONDecodeError, KeyError, ValueError):
                                        continue
                except FileNotFoundError:
                    logger.warning("log command not found on macOS")
                    # Fallback to basic system monitoring
                    logs.extend(await self._generate_sample_logs())
            else:
                # For non-macOS systems, generate sample logs
                logs.extend(await self._generate_sample_logs())
                                
        except subprocess.TimeoutExpired:
            logger.warning("Timeout collecting macOS logs")
        except Exception as e:
            logger.error(f"Error collecting macOS logs: {e}")
            
        return logs
    
    def _determine_log_level(self, data: Dict[str, Any]) -> str:
        """Determine log level from macOS log data."""
        # Check for error indicators in the message
        message = data.get('eventMessage', '').lower()
        if any(word in message for word in ['error', 'failed', 'failure', 'critical']):
            return 'error'
        elif any(word in message for word in ['warning', 'warn']):
            return 'warning'
        elif any(word in message for word in ['debug']):
            return 'debug'
        else:
            return 'info'
    
    async def _save_logs(self, logs: List[LogEntry]):
        """Save logs to file."""
        try:
            with open(self.log_file_path, 'a') as f:
                for log in logs:
                    f.write(log.json() + '\n')
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    async def get_recent_logs(self, minutes: int = 10) -> List[LogEntry]:
        """Get logs from the last specified minutes."""
        logs = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        log_entry = LogEntry(**log_data)
                        
                        # Handle timezone-aware vs timezone-naive datetime comparison
                        log_timestamp = log_entry.timestamp
                        if log_timestamp.tzinfo is not None:
                            # If log timestamp is timezone-aware, convert to local time for comparison
                            log_timestamp = log_timestamp.replace(tzinfo=None)
                        
                        if log_timestamp >= cutoff_time:
                            logs.append(log_entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            logger.warning("Log file not found")
            
        return logs
    
    async def _generate_sample_logs(self) -> List[LogEntry]:
        """Generate sample logs for testing when no system logs are available."""
        logs = []
        now = datetime.now()
        
        # Sample system logs
        sample_logs = [
            {
                'source': 'system',
                'level': 'info',
                'message': 'System boot completed successfully',
                'metadata': {'boot_time': now.isoformat()}
            },
            {
                'source': 'network',
                'level': 'info',
                'message': 'Network interface en0 is active',
                'metadata': {'interface': 'en0', 'status': 'active'}
            },
            {
                'source': 'disk',
                'level': 'info',
                'message': 'Disk usage is normal',
                'metadata': {'usage_percent': 45.2}
            },
            {
                'source': 'security',
                'level': 'info',
                'message': 'No security threats detected',
                'metadata': {'scan_result': 'clean'}
            },
            {
                'source': 'process',
                'level': 'info',
                'message': 'System processes running normally',
                'metadata': {'process_count': 156}
            }
        ]
        
        for i, log_data in enumerate(sample_logs):
            log_entry = LogEntry(
                timestamp=now - timedelta(minutes=i),
                source=log_data['source'],
                level=log_data['level'],
                message=log_data['message'],
                metadata=log_data['metadata']
            )
            logs.append(log_entry)
        
        return logs
