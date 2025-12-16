"""
Profiling utilities for measuring latency and GPU memory usage.
"""
import time
import torch

def start_prof():
    """
    Start profiling: record start time and reset GPU memory stats.
    
    Returns:
        dict: Profiling state (start time, device)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    return {
        "start_time": time.time(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def stop_prof(prof_state):
    """
    Stop profiling: calculate latency and peak GPU memory.
    
    Args:
        prof_state: State returned by start_prof()
    
    Returns:
        tuple: (latency_seconds, gpu_peak_gb)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Calculate latency
    latency = time.time() - prof_state["start_time"]
    
    # Get peak GPU memory (in GB)
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        gpu_peak_gb = peak_memory_bytes / (1024 ** 3)  # Convert to GB
    else:
        gpu_peak_gb = 0.0
    
    return latency, gpu_peak_gb
