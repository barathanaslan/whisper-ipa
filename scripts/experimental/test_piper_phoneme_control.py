
import sys
import wave
from piper import PiperVoice

def main():
    model_path = "piper_test/en_US-lessac-medium.onnx"
    config_path = "piper_test/en_US-lessac-medium.onnx.json"
    
    print(f"Loading voice from {model_path}...")
    voice = PiperVoice.load(model_path, config_path)

    # 1. Get Baseline (US Pronunciation)
    text_us = "tomato"
    # phonemize returns a list of lists (sentences -> phonemes)
    phonemes_us_list = voice.phonemize(text_us) 
    # Flatten just for the first sentence
    phonemes_us = phonemes_us_list[0]
    
    print(f"\nBaseline (US) for '{text_us}':")
    print(f"  Phonemes: {phonemes_us}")
    
    # 2. Create Modified (UK-ish Pronunciation)
    # US: t, ə, m, ˈ, e, ɪ, ɾ, o, ʊ  (approx)
    # UK: t, ə, m, ˈ, ɑ, ː, t, o, ʊ  (approx, manually constructing)
    
    # Let's verify what characters are allowed by printing a few known ones
    # But usually espeak IPA is standard.
    
    # Constructing a manual UK variant
    # Replacing 'e', 'ɪ' (ay) with 'ɑ', 'ː' (ah)
    # Replacing 'ɾ' (flap t) with 't' (true t)
    
    # Note: We must match the config's phoneme map.
    # We can try to reuse the existing list and modify it.
    
    # Copying standard list to modify
    phonemes_uk = list(phonemes_us)
    
    # This is tricky without knowing exact indices, so let's just print US first and Hardcode UK based on standard IPA
    # Piper/espeak uses:
    # US 'tomato':  [t, ə, m, ˈ, e, ɪ, ɾ, o, ʊ]
    # UK 'tomato':  [t, ə, m, ˈ, ɑ, ː, t, o, ʊ]
    
    # Let's try to synthesize both.
    
    # We need to convert phonemes -> ids -> audio
    
    def synthesize_phonemes(phonemes, output_filename):
        print(f"Synthesizing {output_filename} from: {phonemes}")
        # Convert to IDs
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        
        # Synthesize (returns numpy array)
        audio_stream = voice.phoneme_ids_to_audio(phoneme_ids)
        
        # Piper returns a generator? Or array? Docstring said array but let's check.
        # It actually returns a generator of audio chunks usually if using stream, 
        # but the signature scan said -> numpy.ndarray.
        # Let's assume it returns one array or iterable of arrays.
        
        import numpy as np
        
        # If it returns generator, consume it. If array, use it.
        # Based on previous error, it's safer to capture it.
        audio_data = audio_stream
        
        # Convert to int16 for WAV
        # Piper audio is float [-1, 1]
        
        if isinstance(audio_data, np.ndarray):
            pass # good
        else:
            # Maybe it is iterable?
            audio_data = np.concatenate(list(audio_data))
            
        audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(output_filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(audio_int16.tobytes())
            
        print(f"  ✓ Saved to {output_filename}")

    # Generate US Audio
    synthesize_phonemes(phonemes_us, "piper_test/tomato_us.wav")
    
    # Generate UK Audio (Manually constructed list)
    # WARNING: We must ensure these IPA chars are in the model's vocab.
    # 'ɑ' and 'ː' are standard.
    phonemes_uk = ['t', 'ə', 'm', 'ˈ', 'ɑ', 'ː', 't', 'o', 'ʊ']
    
    try:
        synthesize_phonemes(phonemes_uk, "piper_test/tomato_uk.wav")
    except Exception as e:
        print(f"  ✗ Failed to synthesize UK variant: {e}")
        print("    (Maybe a symbol is missing from this voice's map?)")

if __name__ == "__main__":
    main()
