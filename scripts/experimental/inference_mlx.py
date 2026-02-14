import mlx_whisper
import time
import numpy as np
import wave
import os
import subprocess

def create_dummy_audio(filename="dummy_audio.wav", duration=5):
    """Create a dummy sine wave audio file."""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Created dummy audio: {filename}")
    return filename

def get_memory_usage():
    """Get current memory usage of the process."""
    # This is a rough estimate using ps
    pid = os.getpid()
    try:
        res = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(res) / 1024  # MB
    except:
        return 0

def main():
    audio_file = create_dummy_audio()
    
    print("Loading model and warming up...")
    # Load model (this might trigger download)
    # Using 'whisper-large-v3' as requested/implied for full test, or 'tiny' for quick check.
    # User mentioned "getting Whisper... testing it", let's try 'tiny' first to ensure it works, then 'large-v3'.
    # Actually, let's use 'tiny' for the script default to be fast, but I'll add a flag or just use 'tiny' for now.
    # The user wants to see "results", so maybe 'large-v3' is better to see real performance.
    # Let's use 'large-v3' but be aware it might take time to download.
    model_path = "mlx-community/whisper-large-v3-mlx" 
    
    start_mem = get_memory_usage()
    
    start_time = time.time()
    # First run (includes model loading if not lazy, and compilation)
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=model_path)
    end_time = time.time()
    
    end_mem = get_memory_usage()
    
    print(f"Transcription: {result['text']}")
    print(f"Time taken (1st run + load): {end_time - start_time:.2f}s")
    print(f"Memory used (approx): {end_mem - start_mem:.2f} MB")
    
    # Second run (inference only, compiled)
    print("\nRunning second pass (inference only)...")
    start_time = time.time()
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=model_path)
    end_time = time.time()
    
    print(f"Transcription: {result['text']}")
    print(f"Time taken (2nd run): {end_time - start_time:.2f}s")

    # Cleanup
    if os.path.exists(audio_file):
        os.remove(audio_file)

if __name__ == "__main__":
    main()
