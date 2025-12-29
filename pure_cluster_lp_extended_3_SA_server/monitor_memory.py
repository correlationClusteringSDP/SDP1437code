#!/usr/bin/env python3
"""
Monitor memory usage during SDP solving
Can be imported into sdp.py or run separately
"""
import psutil
import os
import sys

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,   # Virtual Memory Size
        'percent': process.memory_percent()
    }

def get_system_memory():
    """Get system memory information"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1024 / 1024 / 1024,
        'available_gb': mem.available / 1024 / 1024 / 1024,
        'used_gb': mem.used / 1024 / 1024 / 1024,
        'percent': mem.percent
    }

def print_memory_status(label=""):
    """Print current memory status"""
    process_mem = get_memory_usage()
    system_mem = get_system_memory()
    
    print(f"\n{'='*60}")
    if label:
        print(f"Memory Status: {label}")
    else:
        print("Memory Status")
    print(f"{'='*60}")
    print(f"Process Memory:")
    print(f"  RSS (Resident Set Size): {process_mem['rss_mb']:.2f} MB")
    print(f"  Virtual Memory Size: {process_mem['vms_mb']:.2f} MB")
    print(f"  Memory Percent: {process_mem['percent']:.2f}%")
    print(f"\nSystem Memory:")
    print(f"  Total: {system_mem['total_gb']:.2f} GB")
    print(f"  Used: {system_mem['used_gb']:.2f} GB ({system_mem['percent']:.1f}%)")
    print(f"  Available: {system_mem['available_gb']:.2f} GB")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Monitor for a specific duration
        import time
        duration = int(sys.argv[1])
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        print(f"Monitoring memory for {duration} seconds (interval: {interval}s)")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            print_memory_status(f"t={int(time.time() - start_time)}s")
            time.sleep(interval)
    else:
        # Just print current status
        print_memory_status()

