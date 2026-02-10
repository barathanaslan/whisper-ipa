"""
Data loader for IPA-transcribed audio datasets.

Loads audio files and their IPA transcriptions, preparing them for
MLX Whisper fine-tuning.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import mlx.core as mx
import mlx_whisper
from mlx_whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


class IPADataset:
    """Dataset class for audio + IPA transcription pairs."""

    def __init__(self, json_path: str, tokenizer, n_mels: int = 80):
        """
        Initialize dataset.

        Args:
            json_path: Path to JSON file with dataset entries
            tokenizer: Whisper tokenizer for encoding IPA text
            n_mels: Number of mel bins (80 for small/medium, 128 for large)
        """
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.n_mels = n_mels

        # Load dataset
        with open(self.json_path) as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {self.json_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        entry = self.data[idx]

        # Load and preprocess audio
        audio_path = entry['audio_path']
        audio = load_audio(audio_path)  # Load and resample to 16kHz

        # Get IPA transcription
        ipa_text = entry['ipa_transcription']

        return {
            'audio': audio,
            'ipa_text': ipa_text,
            'audio_path': audio_path,
            'metadata': {
                'speaker_id': entry.get('speaker_id', 'unknown'),
                'dataset_source': entry.get('dataset_source', 'unknown'),
            }
        }

    def get_batch(self, indices: List[int]) -> Dict:
        """
        Get a batch of samples.

        Args:
            indices: List of sample indices to include in batch

        Returns:
            Dictionary with batched audio features and tokenized IPA text
        """
        samples = [self[i] for i in indices]

        # Process audio: convert to mel spectrograms
        mel_features = []
        for sample in samples:
            audio = sample['audio']
            # Pad or trim to 30 seconds (Whisper's expected input length)
            audio = pad_or_trim(audio)
            # Convert to mel spectrogram with correct n_mels for model size
            mel = log_mel_spectrogram(audio, n_mels=self.n_mels)
            # log_mel_spectrogram returns (n_frames, n_mels) = (3000, n_mels)
            # This is correct format for encoder - DO NOT transpose
            mel_features.append(mel)

        # Stack mel features into batch
        mel_features = mx.array(np.stack(mel_features))

        # Tokenize IPA transcriptions
        # Format: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>[IPA]<|endoftext|>
        ipa_texts = [sample['ipa_text'] for sample in samples]
        tokens = self._tokenize_ipa_batch(ipa_texts)

        return {
            'mel_features': mel_features,  # Shape: (batch_size, n_mels, n_frames)
            'tokens': tokens,              # Shape: (batch_size, seq_len)
            'ipa_texts': ipa_texts,        # Original IPA strings for reference
            'audio_paths': [s['audio_path'] for s in samples],
        }

    def _tokenize_ipa_batch(self, ipa_texts: List[str]) -> mx.array:
        """
        Tokenize a batch of IPA transcriptions with proper special tokens.

        Format: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>[IPA]<|endoftext|>
        """
        tokenized = []

        for ipa_text in ipa_texts:
            # Start with the SOT sequence including notimestamps
            # This gives us: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
            tokens = list(self.tokenizer.sot_sequence_including_notimestamps)

            # Encode the IPA text
            ipa_tokens = self.tokenizer.encode(ipa_text)
            tokens.extend(ipa_tokens)

            # Add end of text token
            tokens.append(self.tokenizer.eot)

            tokenized.append(tokens)

        # Pad sequences to same length
        max_len = max(len(t) for t in tokenized)
        padded = []
        for tokens in tokenized:
            padded_tokens = tokens + [self.tokenizer.eot] * (max_len - len(tokens))
            padded.append(padded_tokens)

        return mx.array(padded)


def create_data_loader(json_path: str, multilingual: bool = True, n_mels: int = 80):
    """
    Create a data loader for the dataset.

    Args:
        json_path: Path to dataset JSON file
        multilingual: Use multilingual tokenizer (better IPA coverage)
        n_mels: Number of mel bins (80 for small/medium, 128 for large)

    Returns:
        IPADataset instance
    """
    from mlx_whisper import tokenizer as tok

    print(f"Loading Whisper tokenizer (multilingual={multilingual})...")
    tokenizer = tok.get_tokenizer(multilingual=multilingual)

    # Set language to English for our English-only dataset
    tokenizer.language = 'en'

    # Create dataset
    dataset = IPADataset(json_path, tokenizer, n_mels=n_mels)

    return dataset


def test_data_loader():
    """Test the data loader with a few samples."""
    print("=" * 60)
    print("Testing IPA Data Loader")
    print("=" * 60)

    # Path to English-only training data
    json_path = "data/processed/english_only_train_ipa.json"

    print(f"\n1. Creating dataset from {json_path}")
    dataset = create_data_loader(json_path, multilingual=True)

    print(f"\n2. Dataset size: {len(dataset)} samples")

    print(f"\n3. Loading first sample...")
    sample = dataset[0]
    print(f"   Audio shape: {sample['audio'].shape}")
    print(f"   IPA text: {sample['ipa_text'][:80]}...")
    print(f"   Source: {sample['metadata']['dataset_source']}")

    print(f"\n4. Creating batch of 4 samples...")
    batch = dataset.get_batch([0, 1, 2, 3])
    print(f"   Mel features shape: {batch['mel_features'].shape}")
    print(f"   Tokens shape: {batch['tokens'].shape}")
    print(f"   IPA texts: {len(batch['ipa_texts'])} transcriptions")

    print("\n" + "=" * 60)
    print("✓ Data loader test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loader()
