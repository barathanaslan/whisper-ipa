#!/usr/bin/env python3
"""
Calculate real training speed from actual process runtime and checkpoints.
This gives an accurate estimate based on actual performance, not theoretical estimates.
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def get_process_runtime(pid):
    """Get process runtime in seconds."""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'etime='],
            capture_output=True,
            text=True,
            check=True
        )
        etime = result.stdout.strip()
        
        # Parse format: [[DD-]hh:]mm:ss
        parts = etime.split(':')
        if len(parts) == 3:  # DD-hh:mm:ss
            days, hours, minutes = parts[0].split('-')
            seconds = parts[2]
            total_sec = int(days) * 86400 + int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        elif len(parts) == 2:  # mm:ss
            minutes, seconds = parts
            total_sec = int(minutes) * 60 + int(seconds)
        else:
            total_sec = int(parts[0])
        
        return total_sec
    except Exception as e:
        print(f"Error getting process runtime: {e}")
        return None

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint step number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.replace('checkpoint-', ''))
    )
    
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    step = int(latest.name.replace('checkpoint-', ''))
    
    # Get checkpoint time
    checkpoint_time = datetime.fromtimestamp(latest.stat().st_mtime)
    
    # Get loss if available
    state_file = latest / "training_state.json"
    loss = None
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            loss = state.get('loss')
    
    return {
        'step': step,
        'path': latest,
        'time': checkpoint_time,
        'loss': loss
    }

def parse_log_for_steps(log_file, num_samples=20):
    """Parse log file for step timing information."""
    log_file = Path(log_file)
    if not log_file.exists():
        return []
    
    steps = []
    try:
        with open(log_file) as f:
            lines = f.readlines()
        
        for line in lines:
            # Look for: "Step 10/20000 | Loss: 2.3456 | Time: 1.234s | Samples/sec: 12.9"
            if "Step" in line and "Time:" in line and "Samples/sec:" in line:
                try:
                    # Extract step number
                    step_part = line.split("Step")[1].split("/")[0].strip()
                    step_num = int(step_part)
                    
                    # Extract time
                    time_part = [p for p in line.split("|") if "Time:" in p][0]
                    time_sec = float(time_part.split("Time:")[1].split("s")[0].strip())
                    
                    # Extract samples/sec
                    samp_part = [p for p in line.split("|") if "Samples/sec:" in p][0]
                    samp_sec = float(samp_part.split("Samples/sec:")[1].strip())
                    
                    steps.append({
                        'step': step_num,
                        'time': time_sec,
                        'samples_per_sec': samp_sec
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return steps[-num_samples:] if steps else []  # Return last N steps

def calculate_real_estimate(pid, checkpoint_dir, total_steps, batch_size=16):
    """Calculate real training estimate based on actual performance."""
    
    print("=" * 70)
    print("Real Training Speed Analysis")
    print("=" * 70)
    print()
    
    # Get process runtime
    runtime_sec = get_process_runtime(pid)
    if runtime_sec is None:
        print(f"❌ Could not get runtime for PID {pid}")
        return
    
    runtime_min = runtime_sec / 60
    runtime_hour = runtime_sec / 3600
    print(f"Process Runtime: {int(runtime_sec//3600)}h {int((runtime_sec%3600)//60)}m {int(runtime_sec%60)}s ({runtime_hour:.2f} hours)")
    print()
    
    # Get latest checkpoint
    checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if checkpoint:
        step = checkpoint['step']
        print(f"Latest Checkpoint: Step {step} / {total_steps}")
        print(f"  Progress: {step * 100 / total_steps:.1f}%")
        if checkpoint['loss']:
            print(f"  Loss: {checkpoint['loss']:.4f}")
        print(f"  Checkpoint Time: {checkpoint['time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Calculate average time per step from runtime
        if step > 0:
            avg_time_per_step = runtime_sec / step
            remaining_steps = total_steps - step
            estimated_remaining_sec = avg_time_per_step * remaining_steps
            estimated_remaining_hour = estimated_remaining_sec / 3600
            
            print(f"Average Time per Step: {avg_time_per_step:.3f} seconds")
            print(f"  (based on {step} steps in {runtime_hour:.2f} hours)")
            print()
            print(f"Estimated Time Remaining: {int(estimated_remaining_sec//3600)}h {int((estimated_remaining_sec%3600)//60)}m ({estimated_remaining_hour:.2f} hours)")
            print(f"Estimated Total Time: {int((runtime_sec + estimated_remaining_sec)//3600)}h {int(((runtime_sec + estimated_remaining_sec)%3600)//60)}m ({(runtime_sec + estimated_remaining_sec)/3600:.2f} hours)")
            print()
            
            # Calculate throughput
            samples_per_sec = batch_size / avg_time_per_step
            print(f"Actual Throughput: {samples_per_sec:.2f} samples/sec")
            print(f"  (batch size {batch_size}, {avg_time_per_step:.3f}s per step)")
    else:
        print("No checkpoints yet - using process runtime to estimate...")
        print()
        print("⚠️  Cannot calculate accurate estimate without checkpoints.")
        print("   First checkpoint will be saved at step 2000.")
        print("   Check back after first checkpoint for accurate estimates.")
        print()
        
        # Very rough estimate: assume we're at step 0-2000
        # This is just a placeholder
        print("Rough estimate (assuming ~1000 steps completed):")
        if runtime_sec > 0:
            rough_steps = min(2000, int(runtime_sec / 2))  # Assume ~2 sec/step
            if rough_steps > 0:
                avg_time_per_step = runtime_sec / rough_steps
                remaining_steps = total_steps - rough_steps
                estimated_remaining_sec = avg_time_per_step * remaining_steps
                print(f"  Estimated time per step: ~{avg_time_per_step:.2f}s")
                print(f"  Estimated remaining: ~{estimated_remaining_sec/3600:.2f} hours")
    
    print()
    
    # Try to parse log file for recent step timings
    log_file = "training_large_log.txt"
    log_steps = parse_log_for_steps(log_file, num_samples=20)
    
    if log_steps:
        print("Recent Step Performance (from logs):")
        print(f"  Last {len(log_steps)} logged steps:")
        
        recent_times = [s['time'] for s in log_steps]
        recent_samp_sec = [s['samples_per_sec'] for s in log_steps]
        
        avg_recent_time = sum(recent_times) / len(recent_times)
        avg_recent_samp_sec = sum(recent_samp_sec) / len(recent_samp_sec)
        
        print(f"  Average step time: {avg_recent_time:.3f}s")
        print(f"  Average throughput: {avg_recent_samp_sec:.2f} samples/sec")
        print()
        
        # Calculate estimate based on recent performance
        latest_logged_step = max(s['step'] for s in log_steps)
        remaining_steps = total_steps - latest_logged_step
        estimated_remaining_sec = avg_recent_time * remaining_steps
        estimated_remaining_hour = estimated_remaining_sec / 3600
        
        print(f"REAL Estimate (based on actual step performance):")
        print(f"  Current step (from logs): {latest_logged_step}")
        print(f"  Estimated Time Remaining: {int(estimated_remaining_sec//3600)}h {int((estimated_remaining_sec%3600)//60)}m ({estimated_remaining_hour:.2f} hours)")
        if checkpoint:
            total_estimated = runtime_sec + estimated_remaining_sec
            print(f"  Estimated Total Time: {int(total_estimated//3600)}h {int((total_estimated%3600)//60)}m ({total_estimated/3600:.2f} hours)")
        else:
            # Estimate total based on recent performance
            total_estimated_sec = avg_recent_time * total_steps
            print(f"  Estimated Total Time: {int(total_estimated_sec//3600)}h {int((total_estimated_sec%3600)//60)}m ({total_estimated_sec/3600:.2f} hours)")
    elif not checkpoint:
        print()
        print("⚠️  No step logs found yet. Training may still be initializing.")
        print("   The training script logs every 10 steps.")
        print("   Run this script again in a few minutes to get real estimates.")
        print()
        print("   Alternatively, wait for the first checkpoint at step 2000")
        print("   (checkpoints are saved every 2000 steps)")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    # Default values from train_large.sh
    PID = 40878
    CHECKPOINT_DIR = "checkpoints/whisper-large-ipa-english"
    TOTAL_STEPS = 20000
    BATCH_SIZE = 16
    
    # Allow override from command line
    if len(sys.argv) > 1:
        PID = int(sys.argv[1])
    if len(sys.argv) > 2:
        CHECKPOINT_DIR = sys.argv[2]
    if len(sys.argv) > 3:
        TOTAL_STEPS = int(sys.argv[3])
    if len(sys.argv) > 4:
        BATCH_SIZE = int(sys.argv[4])
    
    calculate_real_estimate(PID, CHECKPOINT_DIR, TOTAL_STEPS, BATCH_SIZE)

