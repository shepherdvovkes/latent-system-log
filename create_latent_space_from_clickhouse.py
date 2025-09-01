#!/usr/bin/env python3
"""
Create Latent Space from ClickHouse Data
Following best practices for high-performance log analysis
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import json
import pickle
import numpy as np
from loguru import logger

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.latent_space import LatentSpaceService
from app.models.database_models import LatentSpaceDataDB
from app.core.database import AsyncSessionLocal


class ClickHouseLatentSpaceBuilder:
    """Build latent space from ClickHouse data following best practices."""
    
    def __init__(self, clickhouse_host: str = "localhost", clickhouse_port: int = 8123):
        self.clickhouse_url = f"http://{clickhouse_host}:{clickhouse_port}"
        self.database = "system_logs"
        self.table = "raw_logs"
        self.batch_size = 10000
        self.max_logs = 1000000  # Limit for performance
        
    def execute_query(self, query: str, format: str = "JSONEachRow") -> List[Dict[str, Any]]:
        """Execute ClickHouse query and return results."""
        try:
            # Add FORMAT clause for structured output
            if "FORMAT" not in query.upper():
                query += f" FORMAT {format}"
            
            response = requests.post(
                f"{self.clickhouse_url}/",
                params={"query": query},
                headers={"Content-Type": "text/plain"},
                timeout=60
            )
            response.raise_for_status()
            
            if response.text.strip():
                if format == "JSONEachRow":
                    # Parse JSONEachRow format
                    lines = response.text.strip().split('\n')
                    return [json.loads(line) for line in lines if line.strip()]
                else:
                    # For simple queries, return as list
                    return [{"result": response.text.strip()}]
            return []
            
        except Exception as e:
            logger.error(f"ClickHouse query error: {e}")
            return []
    
    def get_logs_for_embeddings(self, limit: int = None) -> List[str]:
        """Extract log messages for embedding generation."""
        if limit is None:
            limit = self.max_logs
            
        query = f"""
        SELECT 
            message,
            timestamp,
            source_type
        FROM {self.database}.{self.table}
        WHERE message != ''
        ORDER BY timestamp
        LIMIT {limit}
        """
        
        logger.info(f"Fetching {limit} logs from ClickHouse...")
        results = self.execute_query(query)
        
        if not results:
            logger.error("No logs retrieved from ClickHouse")
            return []
        
        # Extract messages
        messages = []
        for row in results:
            if isinstance(row, dict):
                message = row.get('message', '')
            else:
                # Handle list format
                message = row[0] if len(row) > 0 else ''
            
            if message and len(message.strip()) > 10:  # Filter out very short messages
                messages.append(message.strip())
        
        logger.info(f"Retrieved {len(messages)} valid log messages")
        return messages
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics from ClickHouse."""
        logger.info("Getting log statistics from ClickHouse...")
        
        stats = {}
        
        # Total count
        total_query = f"SELECT count() as total FROM {self.database}.{self.table}"
        total_result = self.execute_query(total_query)
        stats['total_logs'] = int(total_result[0]['total']) if total_result else 0
        
        # Count by source_type
        source_query = f"""
        SELECT 
            source_type,
            count() as count
        FROM {self.database}.{self.table}
        GROUP BY source_type
        ORDER BY count DESC
        """
        source_result = self.execute_query(source_query)
        stats['by_source_type'] = {row['source_type']: int(row['count']) for row in source_result}
        
        # Date range
        date_query = f"""
        SELECT 
            min(timestamp) as min_date,
            max(timestamp) as max_date
        FROM {self.database}.{self.table}
        """
        date_result = self.execute_query(date_query)
        if date_result:
            stats['date_range'] = {
                'min': date_result[0]['min_date'],
                'max': date_result[0]['max_date']
            }
        
        # Message length statistics
        length_query = f"""
        SELECT 
            avg(length(message)) as avg_length,
            min(length(message)) as min_length,
            max(length(message)) as max_length
        FROM {self.database}.{self.table}
        WHERE message != ''
        """
        length_result = self.execute_query(length_query)
        if length_result:
            stats['message_length'] = {
                'avg': length_result[0]['avg_length'],
                'min': length_result[0]['min_length'],
                'max': length_result[0]['max_length']
            }
        
        return stats
    
    async def build_latent_space(self, max_logs: int = None) -> Dict[str, Any]:
        """Build latent space from ClickHouse data."""
        start_time = time.time()
        
        # Get statistics
        stats = self.get_log_statistics()
        logger.info(f"ClickHouse Statistics: {stats}")
        
        # Get logs for embeddings
        messages = self.get_logs_for_embeddings(max_logs)
        
        if not messages:
            logger.error("No messages to process")
            return {"error": "No messages to process"}
        
        # Initialize latent space service
        latent_space_service = LatentSpaceService()
        await latent_space_service.initialize()
        
        # Build embeddings
        logger.info(f"Building embeddings for {len(messages)} messages...")
        
        # Convert messages to LogEntry objects
        from app.models.schemas import LogEntry
        log_entries = []
        for i, message in enumerate(messages):
            log_entry = LogEntry(
                timestamp=datetime.now(),
                source=f"clickhouse_{i}",
                level="info",
                message=message,
                metadata={"source": "clickhouse"}
            )
            log_entries.append(log_entry)
        
        # Rebuild latent space (this also saves to database)
        logger.info("Rebuilding latent space...")
        await latent_space_service.rebuild_latent_space(log_entries)
        
        # Get final statistics
        final_stats = await self.get_final_statistics()
        
        elapsed_time = time.time() - start_time
        
        result = {
            "success": True,
            "clickhouse_stats": stats,
            "latent_space_stats": final_stats,
            "processing_time": elapsed_time,
            "messages_processed": len(messages),
            "embeddings_created": len(log_entries),
            "save_result": "success"
        }
        
        logger.info(f"Latent space creation completed in {elapsed_time:.2f} seconds")
        return result
    
    async def get_final_statistics(self) -> Dict[str, Any]:
        """Get final latent space statistics."""
        try:
            async with AsyncSessionLocal() as session:
                # Get latest latent space data
                from sqlalchemy import select
                stmt = select(LatentSpaceDataDB).order_by(LatentSpaceDataDB.id.desc()).limit(1)
                result = await session.execute(stmt)
                latest_record = result.scalar_one_or_none()
                
                if latest_record:
                    metadata = latest_record.algorithm_metadata
                    return {
                        "total_embeddings": metadata.get('total_embeddings', 0),
                        "dimension": metadata.get('dimension', 0),
                        "model_name": metadata.get('model_name', 'unknown'),
                        "created_at": latest_record.created_at.isoformat(),
                        "updated_at": latest_record.updated_at.isoformat()
                    }
                else:
                    return {"error": "No latent space data found"}
                    
        except Exception as e:
            logger.error(f"Error getting final statistics: {e}")
            return {"error": str(e)}


async def main():
    """Main function to create latent space from ClickHouse data."""
    print("🚀 Creating Latent Space from ClickHouse Data")
    print("=" * 60)
    
    # Initialize builder
    builder = ClickHouseLatentSpaceBuilder()
    
    # Test ClickHouse connection
    print("🔍 Testing ClickHouse connection...")
    test_query = "SELECT 1 as test"
    test_result = builder.execute_query(test_query)
    
    if not test_result:
        print("❌ Cannot connect to ClickHouse")
        print("💡 Make sure ClickHouse is running and accessible")
        return
    
    print("✅ ClickHouse connection successful")
    
    # Get initial statistics
    print("\n📊 Getting ClickHouse statistics...")
    stats = builder.get_log_statistics()
    
    print(f"📈 Total logs in ClickHouse: {stats.get('total_logs', 0):,}")
    if 'date_range' in stats:
        print(f"📅 Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    
    # Build latent space
    print("\n🔧 Building latent space...")
    result = await builder.build_latent_space(max_logs=500000)  # Process 500K logs for performance
    
    if result.get("success"):
        print("\n🎉 Latent Space Creation Successful!")
        print("=" * 60)
        print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        print(f"📝 Messages processed: {result['messages_processed']:,}")
        print(f"🧠 Embeddings created: {result['embeddings_created']:,}")
        
        if 'latent_space_stats' in result:
            ls_stats = result['latent_space_stats']
            print(f"📊 Embedding dimension: {ls_stats.get('dimension', 'N/A')}")
            print(f"🤖 Model used: {ls_stats.get('model_name', 'N/A')}")
            print(f"📅 Created: {ls_stats.get('created_at', 'N/A')}")
        
        print("\n💡 Next steps:")
        print("   - Use the latent space for similarity search")
        print("   - Query the embeddings for log analysis")
        print("   - Monitor ClickHouse for new logs")
        
    else:
        print(f"❌ Latent space creation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
