"""
Minute log capture service that collects logs every minute and stores them in the database.
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger

from app.services.log_collector import LogCollectorService
from app.services.database_service import DatabaseService
from app.models.schemas import LogEntry


class MinuteLogCaptureService:
    """Service for capturing logs every minute and storing them in the database."""
    
    def __init__(self):
        self.log_collector = LogCollectorService()
        self.database_service = DatabaseService()
        self.is_running = False
        self.scheduler_thread = None
        self.last_capture_info = {
            'timestamp': None,
            'logs_collected': 0,
            'total_logs_in_db': 0,
            'last_interaction_logs': 0
        }
        
    async def initialize(self):
        """Initialize the minute log capture service."""
        logger.info("Initializing MinuteLogCaptureService...")
        
        # Initialize dependencies
        await self.log_collector.initialize()
        await self.database_service.initialize()
        
        # Schedule the log capture task to run every minute
        schedule.every(1).minutes.do(self._capture_and_store_logs)
        
        logger.info("MinuteLogCaptureService initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up MinuteLogCaptureService...")
        self.stop()
        await self.log_collector.cleanup()
        await self.database_service.cleanup()
    
    def start(self):
        """Start the minute log capture service."""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logger.info("MinuteLogCaptureService started")
    
    def stop(self):
        """Stop the minute log capture service."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("MinuteLogCaptureService stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in a separate thread."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(10)  # Check every 10 seconds
    
    async def _capture_and_store_logs(self):
        """Capture logs from the last minute and store them in the database."""
        try:
            logger.info("Starting minute log capture...")
            
            # Collect logs from the last minute
            logs = await self.log_collector.collect_system_logs()
            
            # Filter logs to only include those from the last minute
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            recent_logs = []
            for log in logs:
                # Handle timezone-aware vs timezone-naive datetime comparison
                log_timestamp = log.timestamp
                if log_timestamp.tzinfo is not None:
                    # If log timestamp is timezone-aware, convert to local time for comparison
                    log_timestamp = log_timestamp.replace(tzinfo=None)
                
                if log_timestamp >= one_minute_ago:
                    recent_logs.append(log)
            
            # Store logs in database
            stored_count = 0
            if recent_logs:
                stored_count = await self.database_service.store_logs(recent_logs)
            
            # Get total logs in database from last minute
            total_logs_in_db = len(await self.database_service.get_logs_last_minute())
            
            # Get logs from last interaction (last 30 seconds)
            last_interaction_logs = len(await self._get_last_interaction_logs())
            
            # Update capture info
            self.last_capture_info = {
                'timestamp': datetime.now(),
                'logs_collected': len(recent_logs),
                'total_logs_in_db': total_logs_in_db,
                'last_interaction_logs': last_interaction_logs
            }
            
            logger.info(f"Minute capture completed: {len(recent_logs)} logs collected, {stored_count} stored, {total_logs_in_db} total in DB, {last_interaction_logs} from last interaction")
                
        except Exception as e:
            logger.error(f"Error in minute log capture: {e}")
    
    async def _get_last_interaction_logs(self) -> List[LogEntry]:
        """Get logs from the last 30 seconds (last interaction)."""
        try:
            thirty_seconds_ago = datetime.now() - timedelta(seconds=30)
            from app.core.database import AsyncSessionLocal
            from sqlalchemy import select
            from app.models.database_models import LogEntryDB
            
            async with AsyncSessionLocal() as session:
                
                stmt = select(LogEntryDB).where(
                    LogEntryDB.timestamp >= thirty_seconds_ago
                ).order_by(LogEntryDB.timestamp.desc())
                
                result = await session.execute(stmt)
                db_logs = result.scalars().all()
                
                # Convert to LogEntry objects
                logs = []
                for db_log in db_logs:
                    log_entry = LogEntry(
                        timestamp=db_log.timestamp,
                        source=db_log.source,
                        level=db_log.level,
                        message=db_log.message,
                        metadata=db_log.log_metadata
                    )
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error getting last interaction logs: {e}")
            return []
    
    async def capture_logs_now(self) -> Dict[str, Any]:
        """Manually trigger log capture and return detailed information."""
        try:
            logger.info("Manual log capture triggered...")
            
            # Collect logs
            logs = await self.log_collector.collect_system_logs()
            
            # Filter to last minute
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            recent_logs = []
            for log in logs:
                # Handle timezone-aware vs timezone-naive datetime comparison
                log_timestamp = log.timestamp
                if log_timestamp.tzinfo is not None:
                    # If log timestamp is timezone-aware, convert to local time for comparison
                    log_timestamp = log_timestamp.replace(tzinfo=None)
                
                if log_timestamp >= one_minute_ago:
                    recent_logs.append(log)
            
            # Store logs
            stored_count = 0
            if recent_logs:
                stored_count = await self.database_service.store_logs(recent_logs)
            
            # Get total logs in database from last minute
            total_logs_in_db = len(await self.database_service.get_logs_last_minute())
            
            # Get logs from last interaction
            last_interaction_logs = len(await self._get_last_interaction_logs())
            
            # Update capture info
            self.last_capture_info = {
                'timestamp': datetime.now(),
                'logs_collected': len(recent_logs),
                'total_logs_in_db': total_logs_in_db,
                'last_interaction_logs': last_interaction_logs
            }
            
            logger.info(f"Manual capture: {len(recent_logs)} logs collected, {stored_count} stored, {total_logs_in_db} total in DB, {last_interaction_logs} from last interaction")
            
            return {
                'logs_collected': len(recent_logs),
                'logs_stored': stored_count,
                'total_logs_in_db': total_logs_in_db,
                'last_interaction_logs': last_interaction_logs,
                'timestamp': datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error in manual log capture: {e}")
            return {
                'logs_collected': 0,
                'logs_stored': 0,
                'total_logs_in_db': 0,
                'last_interaction_logs': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def get_last_minute_logs(self) -> List[dict]:
        """Get logs from the last minute from the database."""
        try:
            db_logs = await self.database_service.get_logs_last_minute()
            
            # Convert to dictionary format for API response
            logs = []
            for db_log in db_logs:
                log_dict = {
                    'id': db_log.id,
                    'timestamp': db_log.timestamp.isoformat(),
                    'source': db_log.source,
                    'level': db_log.level,
                    'message': db_log.message,
                    'metadata': db_log.log_metadata,
                    'created_at': db_log.created_at.isoformat() if db_log.created_at else None
                }
                logs.append(log_dict)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting last minute logs: {e}")
            return []
    
    async def get_capture_status(self) -> Dict[str, Any]:
        """Get current capture status and statistics."""
        try:
            # Get current database stats
            total_logs_in_db = len(await self.database_service.get_logs_last_minute())
            last_interaction_logs = len(await self._get_last_interaction_logs())
            
            return {
                'last_capture': self.last_capture_info,
                'current_stats': {
                    'total_logs_in_db': total_logs_in_db,
                    'last_interaction_logs': last_interaction_logs,
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting capture status: {e}")
            return {}
    
    async def get_log_statistics(self, minutes: int = 1) -> dict:
        """Get log statistics for the last N minutes."""
        try:
            return await self.database_service.get_log_statistics(minutes)
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {}
    
    async def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up logs older than specified days."""
        try:
            return await self.database_service.cleanup_old_logs(days)
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            return 0
