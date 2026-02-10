import sys
import mlx.core as mx
from pathlib import Path
import mlx_whisper
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from mlx_whisper.decoding import DecodingOptions, decode
from mlx_whisper.decoding import DecodingOptions, decode

def load_checkpoint_model(checkpoint_path: str, base_model: str = "mlx-community/whisper-large-v3-mlx"):
    print(f"Loading base model architecture: {base_model}")
    model = load_model(base_model)
    model.set_dtype(mx.float32)

    weights_path = Path(checkpoint_path) / "model.safetensors"
    if weights_path.exists():
        print(f"Loading trained weights from: {weights_path}")
        trained_weights = mx.load(str(weights_path))
        
        decoder_weights = {k: v for k, v in trained_weights.items() if k.startswith('decoder.')}
        print(f"Found {len(decoder_weights)} decoder parameters to load")

        from mlx.utils import tree_flatten, tree_unflatten
        current_params = tree_flatten(model.parameters())
        current_params_dict = dict(current_params)

        for k, v in decoder_weights.items():
            if k in current_params_dict:
                current_params_dict[k] = v

        updated_params = tree_unflatten(list(current_params_dict.items()))
        model.update(updated_params)
        mx.eval(model.parameters())
        print("✓ Decoder weights loaded successfully")
    else:
        print(f"ERROR: No weights found at {weights_path}")
        sys.exit(1)

    return model

def transcribe_file(model, audio_path):
    print(f"Transcribing {audio_path}...")
    audio = load_audio(audio_path)
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio, n_mels=128) # Large model uses 128 mels
    mel = mx.expand_dims(mel, 0)
    mel = mel.astype(mx.float32)

    decode_options = DecodingOptions(
        language="en", # We treat IPA as English for tokenizer purposes
        without_timestamps=True,
    )

    audio_features = model.encoder(mel)
    result = decode(model, audio_features, decode_options)
    return result[0].text.strip()

if __name__ == "__main__":
    checkpoint = "checkpoints/whisper-ipa/checkpoint-8000"
    audio_file = "4.wav"
    
    model = load_checkpoint_model(checkpoint)
    transcription = transcribe_file(model, audio_file)
    
    print("\n" + "="*50)
    print(f"Audio: {audio_file}")
    print(f"Prediction: {transcription}")
    print("="*50)
