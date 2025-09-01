"""
System log parser service for processing lastmin.log files.
"""

import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import json

from app.core.database import AsyncSessionLocal
from app.models.system_log_models import (
    SystemLogEntry, LogAnalysis, LogPattern, SystemMetrics, 
    NetworkActivity, ProcessActivity
)
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession


class SystemLogParser:
    """Service for parsing system logs from lastmin.log files."""
    
    def __init__(self):
        self.log_file_path = os.path.join("logs", "lastmin.log")
        self.last_parsed_position = 0
        self.known_patterns = {}
        
    async def parse_log_file(self, minutes: int = 1) -> Dict[str, Any]:
        """Parse the lastmin.log file and store entries in the database."""
        try:
            if not os.path.exists(self.log_file_path):
                logger.warning(f"Log file not found: {self.log_file_path}")
                return {"parsed_entries": 0, "errors": 0}
            
            # Get file size and modification time
            file_stat = os.stat(self.log_file_path)
            current_size = file_stat.st_size
            
            # Read new lines from the file
            new_entries = []
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Skip to last parsed position
                f.seek(self.last_parsed_position)
                
                for line in f:
                    line = line.strip()
                    if line:
                        parsed_entry = self._parse_log_line(line)
                        if parsed_entry:
                            new_entries.append(parsed_entry)
            
            # Update last parsed position
            self.last_parsed_position = current_size
            
            # Store entries in database
            stored_count = 0
            if new_entries:
                stored_count = await self._store_log_entries(new_entries)
                
                # Extract and store additional data
                await self._extract_network_activity(new_entries)
                await self._extract_process_activity(new_entries)
                await self._extract_system_metrics(new_entries)
                
                # Analyze patterns
                await self._analyze_patterns(new_entries)
            
            logger.info(f"Parsed {len(new_entries)} log entries, stored {stored_count}")
            
            return {
                "parsed_entries": len(new_entries),
                "stored_entries": stored_count,
                "errors": 0
            }
            
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
            return {"parsed_entries": 0, "errors": 1, "error": str(e)}
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line from lastmin.log format."""
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
                    'raw_line': line
                }
                
                return {
                    'timestamp': timestamp,
                    'thread_id': thread_id,
                    'log_type': log_type,
                    'activity_id': activity_id,
                    'process_id': int(pid) if pid.isdigit() else None,
                    'ttl': int(ttl) if ttl.isdigit() else None,
                    'source': source,
                    'message': message,
                    'level': level,
                    'log_metadata': log_metadata
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse log line: {line[:100]}... Error: {e}")
            return None
    
    def _determine_log_level(self, message: str) -> str:
        """Determine log level from message content."""
        message_lower = message.lower()
        
        # Error indicators
        if any(word in message_lower for word in ['error', 'failed', 'failure', 'deny', 'violation']):
            return 'error'
        
        # Warning indicators
        if any(word in message_lower for word in ['warning', 'warn', 'deprecated']):
            return 'warning'
        
        # Debug indicators
        if any(word in message_lower for word in ['debug', 'trace']):
            return 'debug'
        
        # Default to info
        return 'info'
    
    async def _store_log_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Store log entries in the database."""
        async with AsyncSessionLocal() as session:
            try:
                db_entries = []
                for entry_data in entries:
                    db_entry = SystemLogEntry(**entry_data)
                    db_entries.append(db_entry)
                
                session.add_all(db_entries)
                await session.commit()
                
                return len(db_entries)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing log entries: {e}")
                return 0
    
    async def _extract_network_activity(self, entries: List[Dict[str, Any]]):
        """Extract network activity from log entries."""
        network_entries = []
        
        for entry in entries:
            message = entry['message']
            
            # TCP connection patterns
            tcp_patterns = [
                r'tcp connect outgoing: \[([^\]]+)\]',
                r'tcp connected: \[([^\]]+)\]',
                r'tcp_connection_summary.*process: ([^(]+)',
                r'udp connect: \[([^\]]+)\]'
            ]
            
            for pattern in tcp_patterns:
                match = re.search(pattern, message)
                if match:
                    connection_info = match.group(1)
                    network_entry = self._parse_network_connection(entry, connection_info)
                    if network_entry:
                        network_entries.append(network_entry)
                    break
        
        if network_entries:
            await self._store_network_activity(network_entries)
    
    def _parse_network_connection(self, log_entry: Dict[str, Any], connection_info: str) -> Optional[Dict[str, Any]]:
        """Parse network connection information."""
        try:
            # Parse connection info like: <IPv4-redacted>:57445<-><IPv4-redacted>:443
            pattern = r'<([^>]+)>:(\d+)<-><([^>]+)>:(\d+)'
            match = re.search(pattern, connection_info)
            
            if match:
                local_addr, local_port, remote_addr, remote_port = match.groups()
                
                # Determine connection type
                connection_type = "TCP"
                if "udp" in log_entry['message'].lower():
                    connection_type = "UDP"
                
                # Extract process name
                process_match = re.search(r'process: ([^(]+)', log_entry['message'])
                process_name = process_match.group(1).strip() if process_match else None
                
                # Extract additional metrics
                duration_match = re.search(r'Duration: ([\d.]+) sec', log_entry['message'])
                duration = float(duration_match.group(1)) if duration_match else None
                
                rtt_match = re.search(r'rtt: ([\d.]+) ms', log_entry['message'])
                rtt = float(rtt_match.group(1)) if rtt_match else None
                
                return {
                    'timestamp': log_entry['timestamp'],
                    'connection_type': connection_type,
                    'local_address': local_addr,
                    'remote_address': remote_addr,
                    'local_port': int(local_port),
                    'remote_port': int(remote_port),
                    'process_name': process_name,
                    'process_id': log_entry['process_id'],
                    'duration': duration,
                    'rtt': rtt,
                    'log_metadata': {'source_log_id': log_entry.get('id')}
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse network connection: {connection_info}, Error: {e}")
            return None
    
    async def _store_network_activity(self, network_entries: List[Dict[str, Any]]):
        """Store network activity in the database."""
        async with AsyncSessionLocal() as session:
            try:
                db_entries = []
                for entry_data in network_entries:
                    db_entry = NetworkActivity(**entry_data)
                    db_entries.append(db_entry)
                
                session.add_all(db_entries)
                await session.commit()
                
                logger.info(f"Stored {len(db_entries)} network activity entries")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing network activity: {e}")
    
    async def _extract_process_activity(self, entries: List[Dict[str, Any]]):
        """Extract process activity from log entries."""
        process_entries = []
        
        for entry in entries:
            message = entry['message']
            source = entry['source']
            
            # Process spawn patterns
            if 'launchd' in source and 'spawned with pid' in message:
                process_entry = self._parse_process_spawn(entry)
                if process_entry:
                    process_entries.append(process_entry)
            
            # Process state changes
            elif 'service state:' in message:
                process_entry = self._parse_process_state_change(entry)
                if process_entry:
                    process_entries.append(process_entry)
        
        if process_entries:
            await self._store_process_activity(process_entries)
    
    def _parse_process_spawn(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse process spawn information."""
        try:
            message = log_entry['message']
            
            # Extract process name and PID
            pattern = r'(\S+)\s+spawned with pid (\d+)'
            match = re.search(pattern, message)
            
            if match:
                process_name = match.group(1)
                process_id = int(match.group(2))
                
                return {
                    'timestamp': log_entry['timestamp'],
                    'process_name': process_name,
                    'process_id': process_id,
                    'activity_type': 'spawn',
                    'log_metadata': {'source_log_id': log_entry.get('id')}
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse process spawn: {log_entry['message']}, Error: {e}")
            return None
    
    def _parse_process_state_change(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse process state change information."""
        try:
            message = log_entry['message']
            
            # Extract state information
            pattern = r'service state: (\S+)'
            match = re.search(pattern, message)
            
            if match:
                state = match.group(1)
                
                # Extract process name from source
                process_name = log_entry['source']
                
                return {
                    'timestamp': log_entry['timestamp'],
                    'process_name': process_name,
                    'process_id': log_entry['process_id'],
                    'activity_type': f'state_change_{state}',
                    'log_metadata': {'state': state, 'source_log_id': log_entry.get('id')}
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse process state change: {log_entry['message']}, Error: {e}")
            return None
    
    async def _store_process_activity(self, process_entries: List[Dict[str, Any]]):
        """Store process activity in the database."""
        async with AsyncSessionLocal() as session:
            try:
                db_entries = []
                for entry_data in process_entries:
                    db_entry = ProcessActivity(**entry_data)
                    db_entries.append(db_entry)
                
                session.add_all(db_entries)
                await session.commit()
                
                logger.info(f"Stored {len(db_entries)} process activity entries")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing process activity: {e}")
    
    async def _extract_system_metrics(self, entries: List[Dict[str, Any]]):
        """Extract system metrics from log entries."""
        metrics_entries = []
        
        for entry in entries:
            message = entry['message']
            
            # CPU usage patterns
            cpu_pattern = r'cpu_usage: ([\d.]+)'
            cpu_match = re.search(cpu_pattern, message)
            if cpu_match:
                metrics_entries.append({
                    'timestamp': entry['timestamp'],
                    'metric_type': 'cpu',
                    'metric_name': 'usage_percent',
                    'metric_value': float(cpu_match.group(1)),
                    'metric_unit': 'percent',
                    'source': entry['source'],
                    'process_id': entry['process_id']
                })
            
            # Memory usage patterns
            memory_pattern = r'memory_usage: ([\d.]+)'
            memory_match = re.search(memory_pattern, message)
            if memory_match:
                metrics_entries.append({
                    'timestamp': entry['timestamp'],
                    'metric_type': 'memory',
                    'metric_name': 'usage_percent',
                    'metric_value': float(memory_match.group(1)),
                    'metric_unit': 'percent',
                    'source': entry['source'],
                    'process_id': entry['process_id']
                })
            
            # Network metrics
            rtt_pattern = r'rtt: ([\d.]+) ms'
            rtt_match = re.search(rtt_pattern, message)
            if rtt_match:
                metrics_entries.append({
                    'timestamp': entry['timestamp'],
                    'metric_type': 'network',
                    'metric_name': 'rtt',
                    'metric_value': float(rtt_match.group(1)),
                    'metric_unit': 'ms',
                    'source': entry['source'],
                    'process_id': entry['process_id']
                })
        
        if metrics_entries:
            await self._store_system_metrics(metrics_entries)
    
    async def _store_system_metrics(self, metrics_entries: List[Dict[str, Any]]):
        """Store system metrics in the database."""
        async with AsyncSessionLocal() as session:
            try:
                db_entries = []
                for entry_data in metrics_entries:
                    db_entry = SystemMetrics(**entry_data)
                    db_entries.append(db_entry)
                
                session.add_all(db_entries)
                await session.commit()
                
                logger.info(f"Stored {len(db_entries)} system metrics entries")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing system metrics: {e}")
    
    async def _analyze_patterns(self, entries: List[Dict[str, Any]]):
        """Analyze log patterns and store them."""
        # This is a basic pattern analysis - can be enhanced with ML
        patterns = {}
        
        for entry in entries:
            message = entry['message']
            source = entry['source']
            
            # Create a simple pattern key
            pattern_key = f"{source}:{self._extract_pattern(message)}"
            
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    'first_seen': entry['timestamp'],
                    'last_seen': entry['timestamp'],
                    'count': 1,
                    'severity': entry['level']
                }
            else:
                patterns[pattern_key]['last_seen'] = entry['timestamp']
                patterns[pattern_key]['count'] += 1
        
        # Store significant patterns
        for pattern_key, pattern_data in patterns.items():
            if pattern_data['count'] >= 3:  # Only store patterns that appear 3+ times
                await self._store_log_pattern(pattern_key, pattern_data)
    
    def _extract_pattern(self, message: str) -> str:
        """Extract a simplified pattern from a message."""
        # Remove specific values but keep structure
        pattern = re.sub(r'\d+', 'N', message)
        pattern = re.sub(r'[a-f0-9]{8,}', 'HASH', pattern)
        pattern = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP', pattern)
        return pattern[:100]  # Limit length
    
    async def _store_log_pattern(self, pattern_key: str, pattern_data: Dict[str, Any]):
        """Store a log pattern in the database."""
        async with AsyncSessionLocal() as session:
            try:
                source, message_pattern = pattern_key.split(':', 1)
                
                db_pattern = LogPattern(
                    pattern_type='frequency',
                    pattern_name=f"Frequent pattern in {source}",
                    pattern_description=f"Pattern appears {pattern_data['count']} times",
                    first_seen=pattern_data['first_seen'],
                    last_seen=pattern_data['last_seen'],
                    occurrence_count=pattern_data['count'],
                    severity=pattern_data['severity'],
                    source_filter=source,
                    message_pattern=message_pattern
                )
                
                session.add(db_pattern)
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing log pattern: {e}")
    
    async def get_log_statistics(self, minutes: int = 1) -> Dict[str, Any]:
        """Get statistics about parsed logs."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now().replace(tzinfo=datetime.now().astimezone().tzinfo) - timedelta(minutes=minutes)
                
                # Get total count
                total_stmt = select(func.count(SystemLogEntry.id)).where(
                    SystemLogEntry.timestamp >= cutoff_time
                )
                total_result = await session.execute(total_stmt)
                total_count = total_result.scalar()
                
                # Get count by level
                level_stats = {}
                for level in ['error', 'warning', 'info', 'debug']:
                    level_stmt = select(func.count(SystemLogEntry.id)).where(
                        SystemLogEntry.level == level,
                        SystemLogEntry.timestamp >= cutoff_time
                    )
                    level_result = await session.execute(level_stmt)
                    level_stats[level] = level_result.scalar()
                
                # Get count by source
                source_stmt = select(SystemLogEntry.source, func.count(SystemLogEntry.id)).where(
                    SystemLogEntry.timestamp >= cutoff_time
                ).group_by(SystemLogEntry.source).order_by(func.count(SystemLogEntry.id).desc()).limit(10)
                
                source_result = await session.execute(source_stmt)
                source_stats = {row[0]: row[1] for row in source_result}
                
                return {
                    'total_logs': total_count,
                    'by_level': level_stats,
                    'by_source': source_stats,
                    'time_range_minutes': minutes
                }
                
            except Exception as e:
                logger.error(f"Error getting log statistics: {e}")
                return {}
