
import sys
import wave
import numpy as np
from piper import PiperVoice

def main():
    model_path = "piper_test/en_US-lessac-medium.onnx"
    config_path = "piper_test/en_US-lessac-medium.onnx.json"
    
    print(f"Loading voice from {model_path}...")
    try:
        voice = PiperVoice.load(model_path, config_path)
    except Exception as e:
        print(f"Failed to load voice: {e}")
        return

    text = "Put the butter on the table."
    
    # helper to synth
    def synth(phonemes, filename):
        print(f"\nSynthesizing: {filename}")
        print(f"Phonemes: {phonemes}")
        ids = voice.phonemes_to_ids(phonemes)
        audio_stream = voice.phoneme_ids_to_audio(ids)
        # Consume stream
        stream = list(audio_stream)
        if len(stream) > 0 and isinstance(stream[0], np.ndarray):
            audio_data = np.concatenate(stream)
        else:
            # Fallback if it's already a single array or scalar
            audio_data = np.array(stream)
            
        # Ensure it's 1D
        audio_data = audio_data.flatten()
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        
        with wave.open(filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(audio_int16.tobytes())
        print(f"✓ Saved {filename}")

    # 1. Baseline
    phonemes_list = voice.phonemize(text)
    baseline_phonemes = phonemes_list[0]
    synth(baseline_phonemes, "piper_test/sentence_baseline.wav")
    
    # 2. Variation: Enunciated T (Replace triggers)
    # Baseline expected: p y t ð ə b ʌ ɾ ɚ ɑ n ð ə t eɪ b l  (using y for u, etc. depending on internal map)
    # Actually let's look at the baseline output first.
    
    # We will manually construct Variation 2 based on the Baseline printed above
    # But for script automation, let's substitute common flap 'ɾ' (if present) with 't'
    
    # Note: Piper/espeak standard IPA might use 'ɾ' (tap) for 'butter'.
    
    variation_phonemes = []
    changes_made = 0
    for p in baseline_phonemes:
        if p == 'ɾ': # Flap T -> True T
            variation_phonemes.append('t')
            changes_made += 1
        else:
            variation_phonemes.append(p)
            
    if changes_made > 0:
        print(f"\nCreating Enunciated Variation ({changes_made} changes)...")
        synth(variation_phonemes, "piper_test/sentence_enunciated.wav")
    else:
        print("\nNo 'ɾ' found to replace. Using manual list check.")
        
    # 3. Variation: Vowel Change (Dialect)
    # 'butter' -> 'b ʊ t ɚ' (using 'ʊ' from 'put')
    # Use baseline again
    # We need to find the index for 'butter' vowels. This is hard programmatically without alignment.
    # So I will inspect the print output first, then maybe update the script or just run a specific hardcoded list if I can guess it.
    
    # Let's try to simulate a heavier change: "Table" -> "Tah-ble" [t ɑ b l]
    # 'eɪ' -> 'ɑ'
    variation_2 = []
    for p in baseline_phonemes:
        if p == 'e': # part of dipthong eɪ
             variation_2.append('ɑ')
        elif p == 'ɪ': # part of dipthong eɪ
             continue # skip
        else:
             variation_2.append(p)
             
    # This is risky blind replacement. 
    # Better approach for this demo:
    # Just print the baseline first, then I will modify the script with exact lists if needed.
    # For now, the Flap T replacement is a safe valid test.

if __name__ == "__main__":
    main()
