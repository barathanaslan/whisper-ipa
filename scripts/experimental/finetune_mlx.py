import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_whisper
import time
import numpy as np

def train_step(model, optimizer, audio, tokens):
    def loss_fn(model, audio, tokens):
        # Forward pass
        # Note: Whisper's forward typically takes audio_features and tokens
        # We need to check how mlx_whisper model expects input.
        # Usually: logits = model(audio_features, tokens)
        # But we need to compute features first or pass raw audio if model handles it.
        # mlx_whisper.transcribe handles everything, but the model itself might be lower level.
        
        # Let's assume model is the Whisper model class
        # We might need to compute log_spec manually if the model expects it.
        # For this PoC, we will try to pass features.
        
        logits = model(audio, tokens)
        
        # Simple cross entropy
        # tokens is [batch, seq_len], logits is [batch, seq_len, vocab]
        # We ignore the last token prediction for the loss usually, or shift.
        # For PoC, just compute loss on all.
        
        # Flatten
        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        tokens_flat = tokens.reshape(-1)
        
        loss = nn.losses.cross_entropy(logits_flat, tokens_flat)
        return mx.mean(loss)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, audio, tokens)
    optimizer.update(model, grads)
    return loss

def main():
    print("Loading model for fine-tuning...")
    # We need to load the underlying model, not just the pipeline.
    # mlx_whisper.load_models might have what we need.
    try:
        # Attempt to find the load function
        if hasattr(mlx_whisper, "load_model"):
            model = mlx_whisper.load_model("mlx-community/whisper-large-v3-mlx")
        elif hasattr(mlx_whisper.load_models, "load_model"):
            model = mlx_whisper.load_models.load_model("mlx-community/whisper-large-v3-mlx")
        else:
            print("Could not find load_model")
            return
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Model loaded.")
    
    # Freeze layers? 
    # For PoC, let's just train the whole thing or last layer.
    # model.freeze() # if available, or manually set trainable.
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-5)
    
    # Dummy Data
    # Batch size 1
    # Audio features: [1, 3000, 128] (128 mel bins for large-v3, 3000 frames)
    # MLX Conv1d expects (N, L, C)
    audio_features = mx.random.normal((1, 3000, 128))
    tokens = mx.random.randint(0, 50000, (1, 50))
    
    print("Starting training loop...")
    start_time = time.time()
    
    for i in range(5):
        step_start = time.time()
        loss = train_step(model, optimizer, audio_features, tokens)
        mx.eval(loss) # Force computation
        step_time = time.time() - step_start
        print(f"Step {i+1}: Loss={loss.item():.4f}, Time={step_time:.4f}s")
        
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    
if __name__ == "__main__":
    main()
