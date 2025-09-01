#!/usr/bin/env python3
"""
Script to create latent space data from existing logs in the database.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import AsyncSessionLocal, init_db
from app.models.database_models import LogEntryDB
from app.services.latent_space import LatentSpaceService
from app.models.schemas import LogEntry
from sqlalchemy import select
from datetime import datetime, timedelta
from loguru import logger


async def create_latent_space_data():
    """Create latent space data from existing logs."""
    logger.info("Starting latent space data creation...")
    
    try:
        # Initialize database
        await init_db()
        
        # Initialize latent space service
        latent_space_service = LatentSpaceService()
        await latent_space_service.initialize()
        
        # Get logs from database
        async with AsyncSessionLocal() as session:
            # Get recent logs (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            stmt = select(LogEntryDB).where(
                LogEntryDB.timestamp >= cutoff_time
            ).order_by(LogEntryDB.timestamp.desc())
            
            result = await session.execute(stmt)
            db_logs = result.scalars().all()
            
            logger.info(f"Found {len(db_logs)} logs from the last 24 hours")
            
            if not db_logs:
                logger.warning("No logs found in the last 24 hours, getting all logs")
                stmt = select(LogEntryDB).order_by(LogEntryDB.timestamp.desc()).limit(1000)
                result = await session.execute(stmt)
                db_logs = result.scalars().all()
                logger.info(f"Found {len(db_logs)} total logs")
            
            # Convert to LogEntry format
            logs = []
            for db_log in db_logs:
                log = LogEntry(
                    timestamp=db_log.timestamp,
                    source=db_log.source,
                    level=db_log.level,
                    message=db_log.message,
                    metadata=db_log.log_metadata
                )
                logs.append(log)
            
            logger.info(f"Converted {len(logs)} logs to LogEntry format")
            
            # Rebuild latent space with the logs
            logger.info("Rebuilding latent space...")
            await latent_space_service.rebuild_latent_space(logs)
            
            logger.info("Latent space data creation completed successfully!")
            
            # Show statistics
            stats = await latent_space_service.get_latent_space_stats()
            logger.info(f"Latent space statistics:")
            logger.info(f"  - Total embeddings: {stats.get('total_embeddings', 0)}")
            logger.info(f"  - Model: {stats.get('model_name', 'Unknown')}")
            logger.info(f"  - Dimension: {stats.get('embedding_dimension', 0)}")
            logger.info(f"  - Index size: {stats.get('index_size', 0)}")
            
    except Exception as e:
        logger.error(f"Error creating latent space data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")


async def show_latent_space_data():
    """Show current latent space data."""
    logger.info("Showing latent space data...")
    
    try:
        # Initialize database
        await init_db()
        
        # Initialize latent space service
        latent_space_service = LatentSpaceService()
        await latent_space_service.initialize()
        
        # Get statistics
        stats = await latent_space_service.get_latent_space_stats()
        
        print("\n" + "="*60)
        print("LATENT SPACE DATA STATUS")
        print("="*60)
        print(f"Initialized: {stats.get('is_initialized', False)}")
        print(f"Total Embeddings: {stats.get('total_embeddings', 0)}")
        print(f"Model Name: {stats.get('model_name', 'Unknown')}")
        print(f"Dimension: {stats.get('embedding_dimension', 0)}")
        print(f"Index Size: {stats.get('index_size', 0)}")
        print(f"Memory Usage: {stats.get('memory_usage_mb', 0):.2f} MB")
        print(f"Last Updated: {stats.get('last_updated', 'Never')}")
        
        if stats.get('sources_distribution'):
            print(f"\nSources Distribution:")
            for source, count in stats['sources_distribution'].items():
                print(f"  - {source}: {count}")
        
        if stats.get('levels_distribution'):
            print(f"\nLevels Distribution:")
            for level, count in stats['levels_distribution'].items():
                print(f"  - {level}: {count}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error showing latent space data: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        asyncio.run(show_latent_space_data())
    else:
        asyncio.run(create_latent_space_data())
