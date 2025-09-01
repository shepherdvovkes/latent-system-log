"""
Database service for log storage operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import AsyncSessionLocal
from app.models.database_models import LogEntryDB
from app.models.schemas import LogEntry


class DatabaseService:
    """Service for database operations related to log storage."""
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
        
    async def initialize(self):
        """Initialize the database service."""
        logger.info("Initializing DatabaseService...")
        from app.core.database import init_db
        await init_db()
        logger.info("DatabaseService initialized successfully")
    
    async def cleanup(self):
        """Cleanup database resources."""
        logger.info("Cleaning up DatabaseService...")
        from app.core.database import close_db
        await close_db()
    
    async def store_logs(self, logs: List[LogEntry]) -> int:
        """Store log entries in the database."""
        if not logs:
            return 0
            
        async with AsyncSessionLocal() as session:
            try:
                # Convert LogEntry objects to LogEntryDB objects
                db_logs = []
                for log in logs:
                    db_log = LogEntryDB(
                        timestamp=log.timestamp,
                        source=log.source,
                        level=log.level,
                        message=log.message,
                        log_metadata=log.metadata
                    )
                    db_logs.append(db_log)
                
                # Bulk insert
                session.add_all(db_logs)
                await session.commit()
                
                logger.info(f"Stored {len(db_logs)} log entries in database")
                return len(db_logs)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing logs in database: {e}")
                raise
    
    async def get_logs_last_minute(self) -> List[LogEntryDB]:
        """Get logs from the last minute."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=1)
                
                stmt = select(LogEntryDB).where(
                    LogEntryDB.timestamp >= cutoff_time
                ).order_by(LogEntryDB.timestamp.desc())
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                logger.info(f"Retrieved {len(logs)} logs from the last minute")
                return logs
                
            except Exception as e:
                logger.error(f"Error retrieving logs from database: {e}")
                return []
    
    async def get_logs_by_time_range(self, start_time: datetime, end_time: datetime) -> List[LogEntryDB]:
        """Get logs within a specific time range."""
        async with AsyncSessionLocal() as session:
            try:
                stmt = select(LogEntryDB).where(
                    LogEntryDB.timestamp >= start_time,
                    LogEntryDB.timestamp <= end_time
                ).order_by(LogEntryDB.timestamp.desc())
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                logger.info(f"Retrieved {len(logs)} logs from {start_time} to {end_time}")
                return logs
                
            except Exception as e:
                logger.error(f"Error retrieving logs by time range: {e}")
                return []
    
    async def get_logs_by_source(self, source: str, minutes: int = 1) -> List[LogEntryDB]:
        """Get logs from a specific source within the last N minutes."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                
                stmt = select(LogEntryDB).where(
                    LogEntryDB.source == source,
                    LogEntryDB.timestamp >= cutoff_time
                ).order_by(LogEntryDB.timestamp.desc())
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                logger.info(f"Retrieved {len(logs)} logs from source '{source}' in the last {minutes} minutes")
                return logs
                
            except Exception as e:
                logger.error(f"Error retrieving logs by source: {e}")
                return []
    
    async def get_logs_by_level(self, level: str, minutes: int = 1) -> List[LogEntryDB]:
        """Get logs of a specific level within the last N minutes."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                
                stmt = select(LogEntryDB).where(
                    LogEntryDB.level == level,
                    LogEntryDB.timestamp >= cutoff_time
                ).order_by(LogEntryDB.timestamp.desc())
                
                result = await session.execute(stmt)
                logs = result.scalars().all()
                
                logger.info(f"Retrieved {len(logs)} {level} logs in the last {minutes} minutes")
                return logs
                
            except Exception as e:
                logger.error(f"Error retrieving logs by level: {e}")
                return []
    
    async def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up logs older than specified days."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now() - timedelta(days=days)
                
                stmt = delete(LogEntryDB).where(LogEntryDB.timestamp < cutoff_time)
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old log entries")
                return deleted_count
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error cleaning up old logs: {e}")
                return 0
    
    async def get_log_statistics(self, minutes: int = 1) -> dict:
        """Get log statistics for the last N minutes."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                
                # Get total count
                total_stmt = select(LogEntryDB).where(LogEntryDB.timestamp >= cutoff_time)
                total_result = await session.execute(total_stmt)
                total_count = len(total_result.scalars().all())
                
                # Get count by level
                level_stats = {}
                for level in ['debug', 'info', 'warning', 'error', 'critical']:
                    level_stmt = select(LogEntryDB).where(
                        LogEntryDB.level == level,
                        LogEntryDB.timestamp >= cutoff_time
                    )
                    level_result = await session.execute(level_stmt)
                    level_stats[level] = len(level_result.scalars().all())
                
                # Get count by source
                source_stmt = select(LogEntryDB.source).where(LogEntryDB.timestamp >= cutoff_time)
                source_result = await session.execute(source_stmt)
                sources = source_result.scalars().all()
                source_stats = {}
                for source in set(sources):
                    source_count_stmt = select(LogEntryDB).where(
                        LogEntryDB.source == source,
                        LogEntryDB.timestamp >= cutoff_time
                    )
                    source_count_result = await session.execute(source_count_stmt)
                    source_stats[source] = len(source_count_result.scalars().all())
                
                stats = {
                    'total_logs': total_count,
                    'by_level': level_stats,
                    'by_source': source_stats,
                    'time_range_minutes': minutes
                }
                
                return stats
                
            except Exception as e:
                logger.error(f"Error getting log statistics: {e}")
                return {}
