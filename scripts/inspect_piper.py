
from piper import PiperVoice
import inspect

print("Inspecting PiperVoice.synthesize signature:")
print(inspect.signature(PiperVoice.synthesize))
print("\nDocstring:")
print(PiperVoice.synthesize.__doc__)
