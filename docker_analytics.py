#!/usr/bin/env python3
"""
Docker-based ClickHouse Analytics for System Logs
High-performance log analysis using Docker containers
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd


class DockerClickHouseAnalytics:
    """High-performance analytics for system logs using Docker-based ClickHouse."""
    
    def __init__(self, host: str = "localhost", port: int = 8123):
        self.base_url = f"http://{host}:{port}"
        self.database = "system_logs"
        self.table = "raw_logs"
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a ClickHouse query and return results."""
        try:
            response = requests.post(
                f"{self.base_url}/",
                params={"query": query},
                headers={"Content-Type": "text/plain"},
                timeout=30
            )
            response.raise_for_status()
            
            # Parse JSON response
            if response.text.strip():
                return response.json()
            return []
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []
    
    def get_total_count(self) -> int:
        """Get total number of log entries."""
        query = f"SELECT count() as total FROM {self.database}.{self.table}"
        result = self.execute_query(query)
        return result[0]['total'] if result else 0
    
    def get_logs_by_level(self) -> Dict[str, int]:
        """Get log count by level."""
        query = f"""
        SELECT 
            level,
            count() as count
        FROM {self.database}.{self.table}
        GROUP BY level
        ORDER BY count DESC
        """
        result = self.execute_query(query)
        return {row['level']: row['count'] for row in result}
    
    def get_top_sources(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top log sources."""
        query = f"""
        SELECT 
            source,
            count() as count,
            countIf(level = 'error') as errors,
            countIf(level = 'warning') as warnings
        FROM {self.database}.{self.table}
        GROUP BY source
        ORDER BY count DESC
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_logs_by_hour(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get log count by hour for the last N hours."""
        query = f"""
        SELECT 
            toStartOfHour(timestamp) as hour,
            count() as count,
            countIf(level = 'error') as errors,
            countIf(level = 'warning') as warnings
        FROM {self.database}.{self.table}
        WHERE timestamp >= now() - INTERVAL {hours} HOUR
        GROUP BY hour
        ORDER BY hour
        """
        return self.execute_query(query)
    
    def get_error_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get error trends over time."""
        query = f"""
        SELECT 
            toStartOfMinute(timestamp) as minute,
            count() as error_count,
            uniq(source) as unique_sources
        FROM {self.database}.{self.table}
        WHERE level = 'error' 
          AND timestamp >= now() - INTERVAL {hours} HOUR
        GROUP BY minute
        ORDER BY minute
        """
        return self.execute_query(query)
    
    def search_logs(self, 
                   search_term: str = None,
                   source: str = None,
                   level: str = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs with filters."""
        conditions = []
        
        if search_term:
            conditions.append(f"message ILIKE '%{search_term}%'")
        if source:
            conditions.append(f"source = '{source}'")
        if level:
            conditions.append(f"level = '{level}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT 
            timestamp,
            source,
            level,
            message,
            pid,
            thread_id
        FROM {self.database}.{self.table}
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        # Query execution time
        start_time = time.time()
        total_count = self.get_total_count()
        query_time = time.time() - start_time
        
        # Get recent activity
        recent_query = f"""
        SELECT 
            count() as recent_count,
            uniq(source) as active_sources
        FROM {self.database}.{self.table}
        WHERE timestamp >= now() - INTERVAL 1 HOUR
        """
        recent_result = self.execute_query(recent_query)
        recent_data = recent_result[0] if recent_result else {}
        
        return {
            "total_logs": total_count,
            "query_time_seconds": query_time,
            "recent_logs_1h": recent_data.get('recent_count', 0),
            "active_sources_1h": recent_data.get('active_sources', 0),
            "database": self.database,
            "table": self.table
        }
    
    def get_docker_status(self) -> Dict[str, Any]:
        """Get Docker container status."""
        import subprocess
        
        try:
            # Check if containers are running
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            return {
                "containers": containers,
                "total_containers": len(containers),
                "running_containers": len([c for c in containers if c.get('State') == 'running'])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def export_to_csv(self, query: str, filename: str):
        """Export query results to CSV."""
        result = self.execute_query(query)
        if result:
            df = pd.DataFrame(result)
            df.to_csv(filename, index=False)
            print(f"✅ Exported {len(result)} rows to {filename}")
        else:
            print("❌ No data to export")


def main():
    """Main function to demonstrate Docker-based ClickHouse analytics."""
    print("🚀 Docker-based ClickHouse Analytics for System Logs")
    print("=" * 60)
    
    # Initialize analytics
    analytics = DockerClickHouseAnalytics()
    
    # Check Docker status
    print("🐳 Docker Status:")
    docker_status = analytics.get_docker_status()
    if "error" in docker_status:
        print(f"   ❌ Docker error: {docker_status['error']}")
    else:
        print(f"   📦 Total containers: {docker_status['total_containers']}")
        print(f"   ✅ Running containers: {docker_status['running_containers']}")
    
    # Check connection
    try:
        total_count = analytics.get_total_count()
        print(f"\n📊 Total logs in database: {total_count:,}")
    except Exception as e:
        print(f"❌ Cannot connect to ClickHouse: {e}")
        print("💡 Make sure Docker stack is running: ./start_docker_stack.sh")
        return
    
    # Performance metrics
    print("\n📈 Performance Metrics:")
    metrics = analytics.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if value > 1000:
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    # Log levels
    print("\n📊 Log Levels:")
    levels = analytics.get_logs_by_level()
    for level, count in levels.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"   {level}: {count:,} ({percentage:.1f}%)")
    
    # Top sources
    print("\n🔝 Top Log Sources:")
    sources = analytics.get_top_sources(5)
    for source in sources:
        print(f"   {source['source']}: {source['count']:,} logs")
    
    # Recent activity
    print("\n⏰ Recent Activity (Last Hour):")
    hourly = analytics.get_logs_by_hour(1)
    if hourly:
        for hour_data in hourly:
            hour = hour_data['hour']
            count = hour_data['count']
            errors = hour_data['errors']
            warnings = hour_data['warnings']
            print(f"   {hour}: {count:,} logs ({errors} errors, {warnings} warnings)")
    
    # Search example
    print("\n🔍 Recent Error Logs:")
    errors = analytics.search_logs(level="error", limit=3)
    for error in errors:
        timestamp = error['timestamp']
        source = error['source']
        message = error['message'][:100] + "..." if len(error['message']) > 100 else error['message']
        print(f"   {timestamp} [{source}]: {message}")
    
    print("\n" + "=" * 60)
    print("🎉 Docker-based ClickHouse Analytics Complete!")
    print("💡 Try these commands:")
    print("   - View logs: docker-compose logs -f")
    print("   - Stop services: docker-compose down")
    print("   - Restart: docker-compose restart")


if __name__ == "__main__":
    main()
