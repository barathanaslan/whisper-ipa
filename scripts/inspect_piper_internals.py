
from piper import PiperVoice

model_path = "piper_test/en_US-lessac-medium.onnx"
config_path = "piper_test/en_US-lessac-medium.onnx.json"
voice = PiperVoice.load(model_path, config_path)

print("Attributes of PiperVoice:")
for attr in dir(voice):
    if not attr.startswith("__"):
        print(f"  {attr}")
