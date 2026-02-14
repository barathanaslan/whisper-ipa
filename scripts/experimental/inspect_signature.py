
from piper import PiperVoice
import inspect

print("Inspecting PiperVoice.phoneme_ids_to_audio signature:")
print(inspect.signature(PiperVoice.phoneme_ids_to_audio))
print("\nDocstring:")
print(PiperVoice.phoneme_ids_to_audio.__doc__)
