#!/usr/bin/env python3
"""
Initialize system log database tables.
"""

import asyncio
from app.core.database import init_db
from app.models.system_log_models import (
    SystemLogEntry, LogAnalysis, LogPattern, SystemMetrics, 
    NetworkActivity, ProcessActivity
)

async def init_system_log_database():
    """Initialize the system log database tables."""
    print("Initializing system log database...")
    
    try:
        await init_db()
        print("System log database initialized successfully!")
        
        # Test the database connection
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = result.fetchall()
            print(f"Available tables: {[table[0] for table in tables]}")
            
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    asyncio.run(init_system_log_database())
