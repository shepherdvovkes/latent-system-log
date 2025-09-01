#!/usr/bin/env python3
"""
Analyze filtered lines to understand what was excluded during import.
"""

import re
import os
from collections import Counter
from datetime import datetime


def analyze_filtered_lines(log_file: str, sample_size: int = 10000):
    """Analyze what lines were filtered out during import."""
    print(f"🔍 Analyzing filtered lines in: {log_file}")
    print(f"📏 File size: {os.path.getsize(log_file) / (1024*1024*1024):.2f} GB")
    print("="*80)
    
    # Pattern used in the importer
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
    
    total_lines = 0
    matched_lines = 0
    filtered_lines = 0
    empty_lines = 0
    header_lines = 0
    malformed_samples = []
    line_types = Counter()
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num >= sample_size:
                break
                
            total_lines += 1
            original_line = line
            line = line.strip()
            
            if not line:
                empty_lines += 1
                line_types['empty'] += 1
                continue
            
            # Check if it's a header line
            if line.startswith('Timestamp') and 'Thread' in line and 'Type' in line:
                header_lines += 1
                line_types['header'] += 1
                continue
            
            # Try to match the pattern
            match = re.match(pattern, line)
            if match:
                matched_lines += 1
                line_types['valid'] += 1
            else:
                filtered_lines += 1
                line_types['malformed'] += 1
                
                # Collect samples of malformed lines
                if len(malformed_samples) < 10:
                    malformed_samples.append({
                        'line_num': line_num + 1,
                        'line': line[:200] + '...' if len(line) > 200 else line,
                        'length': len(line)
                    })
    
    # Calculate statistics
    match_rate = (matched_lines / total_lines * 100) if total_lines > 0 else 0
    filter_rate = (filtered_lines / total_lines * 100) if total_lines > 0 else 0
    
    print(f"📊 ANALYSIS RESULTS (Sample of {sample_size:,} lines):")
    print(f"   Total Lines Analyzed: {total_lines:,}")
    print(f"   ✅ Matched Format: {matched_lines:,} ({match_rate:.1f}%)")
    print(f"   ❌ Filtered Out: {filtered_lines:,} ({filter_rate:.1f}%)")
    print(f"   📄 Empty Lines: {empty_lines:,}")
    print(f"   🏷️  Header Lines: {header_lines:,}")
    
    print(f"\n📈 LINE TYPE BREAKDOWN:")
    for line_type, count in line_types.most_common():
        percentage = (count / total_lines * 100) if total_lines > 0 else 0
        print(f"   {line_type}: {count:,} ({percentage:.1f}%)")
    
    if malformed_samples:
        print(f"\n🔍 SAMPLE MALFORMED LINES:")
        for i, sample in enumerate(malformed_samples, 1):
            print(f"   {i}. Line {sample['line_num']}:")
            print(f"      Length: {sample['length']} chars")
            print(f"      Content: {sample['line']}")
            print()
    
    # Estimate total filtered lines
    estimated_total_filtered = (filtered_lines / sample_size) * 20280453
    estimated_total_matched = (matched_lines / sample_size) * 20280453
    
    print(f"📊 ESTIMATED FULL FILE ANALYSIS:")
    print(f"   Estimated Total Lines: 20,280,453")
    print(f"   Estimated Matched: {estimated_total_matched:,.0f}")
    print(f"   Estimated Filtered: {estimated_total_filtered:,.0f}")
    print(f"   Actual Stored: 17,473,986")
    print(f"   Difference: {estimated_total_matched - 17473986:,.0f}")
    
    print("\n" + "="*80)
    
    if match_rate > 80:
        print("✅ Filtering appears to be working correctly - high match rate!")
    else:
        print("⚠️  Low match rate - may need to adjust parsing logic")


def analyze_line_patterns(log_file: str, sample_size: int = 5000):
    """Analyze common patterns in filtered lines."""
    print(f"\n🔍 ANALYZING LINE PATTERNS (Sample of {sample_size:,} lines)")
    print("="*80)
    
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.+)$'
    
    filtered_patterns = Counter()
    timestamp_patterns = Counter()
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num >= sample_size:
                break
                
            line = line.strip()
            if not line or line.startswith('Timestamp'):
                continue
            
            # Check if it matches the pattern
            match = re.match(pattern, line)
            if not match:
                # Analyze why it doesn't match
                parts = line.split()
                
                # Check timestamp format
                if parts:
                    timestamp_part = parts[0]
                    if re.match(r'^\d{4}-\d{2}-\d{2}', timestamp_part):
                        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{4}$', timestamp_part):
                            timestamp_patterns['valid_timestamp'] += 1
                        else:
                            timestamp_patterns['invalid_timestamp_format'] += 1
                    else:
                        timestamp_patterns['no_timestamp'] += 1
                
                # Count parts
                if len(parts) < 7:
                    filtered_patterns[f'too_few_parts_{len(parts)}'] += 1
                elif len(parts) > 7:
                    filtered_patterns[f'too_many_parts_{len(parts)}'] += 1
                else:
                    filtered_patterns['wrong_format_7_parts'] += 1
    
    print(f"📊 TIMESTAMP ANALYSIS:")
    for pattern_type, count in timestamp_patterns.most_common():
        print(f"   {pattern_type}: {count:,}")
    
    print(f"\n📊 PART COUNT ANALYSIS:")
    for pattern_type, count in filtered_patterns.most_common():
        print(f"   {pattern_type}: {count:,}")


if __name__ == "__main__":
    log_file = "lastday.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
    else:
        analyze_filtered_lines(log_file, sample_size=50000)
        analyze_line_patterns(log_file, sample_size=10000)
