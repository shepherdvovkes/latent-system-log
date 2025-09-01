#!/usr/bin/env python3
"""
Preview script to check the format of lastday.log before bulk import.
"""

import os
import re
from datetime import datetime


def preview_log_format(log_file: str, num_lines: int = 10):
    """Preview the first few lines of the log file to verify format."""
    print(f"🔍 Previewing log file: {log_file}")
    print(f"📏 File size: {os.path.getsize(log_file) / (1024*1024*1024):.2f} GB")
    print("="*80)
    
    # Pattern to match the expected format
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
    
    matched_lines = 0
    total_lines = 0
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
                
            line = line.strip()
            if line:
                total_lines += 1
                match = re.match(pattern, line)
                
                if match:
                    matched_lines += 1
                    timestamp_str, thread_id, log_type, activity_id, pid, ttl, rest = match.groups()
                    
                    # Parse timestamp
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f%z')
                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                    
                    # Split source and message
                    if ': ' in rest:
                        source, message = rest.split(': ', 1)
                    else:
                        source = rest
                        message = ""
                    
                    print(f"✅ Line {i+1}:")
                    print(f"   Timestamp: {timestamp_str}")
                    print(f"   Thread: {thread_id}")
                    print(f"   Type: {log_type}")
                    print(f"   Activity: {activity_id}")
                    print(f"   PID: {pid}")
                    print(f"   TTL: {ttl}")
                    print(f"   Source: {source}")
                    print(f"   Message: {message[:100]}{'...' if len(message) > 100 else ''}")
                    print()
                else:
                    print(f"❌ Line {i+1} (unmatched format):")
                    print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
                    print()
    
    print("="*80)
    print(f"📊 Format Analysis:")
    print(f"   Total lines checked: {total_lines}")
    print(f"   Matched format: {matched_lines}")
    print(f"   Unmatched: {total_lines - matched_lines}")
    print(f"   Match rate: {matched_lines/total_lines*100:.1f}%" if total_lines > 0 else "N/A")
    
    if matched_lines > 0:
        print("✅ Log format appears to be compatible with the bulk importer!")
    else:
        print("❌ Log format may not be compatible. Please check the format.")


if __name__ == "__main__":
    log_file = "lastday.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
    else:
        preview_log_format(log_file, num_lines=5)
