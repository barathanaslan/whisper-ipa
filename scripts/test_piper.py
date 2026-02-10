
import sys
import wave
import json
from pathlib import Path

try:
    from piper import PiperVoice
except ImportError:
    print("CRITICAL: piper package not found. Installation likely incomplete.")
    sys.exit(1)

def main():
    model_path = "piper_test/en_US-lessac-medium.onnx"
    config_path = "piper_test/en_US-lessac-medium.onnx.json"
    output_wav = "piper_test/test_output.wav"
    text = "Hello, world! This is a test of phonetic extraction."

    print(f"Loading voice from {model_path}...")
    try:
        # Load voice
        voice = PiperVoice.load(model_path, config_path)
        print("✓ Voice loaded successfully")

        # Test Phonemization (The Oracle Step)
        print(f"\nInput Text: '{text}'")
        print("Phonemizing...")
        # Note: Piper's python API usually exposes .phonemize()
        phonemes = voice.phonemize(text)
        print(f"Generated Phonemes: {phonemes}")
        
        # Verify it's a list/generator of phoneme strings or ids
        phoneme_str = "".join(str(p) for p in phonemes)
        print(f"Phoneme String: {phoneme_str}")

        # Test Synthesis
        print("\nSynthesizing audio...")
        with wave.open(output_wav, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050) # Matching the model config
            voice.synthesize(text, wav_file)
            
        print(f"✓ Audio saved to {output_wav}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
