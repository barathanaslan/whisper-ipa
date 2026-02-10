import mlx_whisper.tokenizer as tok

def check_tokens():
    tokenizer = tok.get_tokenizer(multilingual=True)
    
    # The tricky Mandarin string
    mandarin_ipa = "tɕiɑŋ˨˩˦ joʊ˨˩˦ puɔ˧˥ peɪ˥˩ phaɪ˥ tɑʊ˥˩ taɪ˥˩ ʈʂɤ˨˩˦ ny˨˩˦ ɚ˧˥ ʈʂhu˥ joʊ˧˥"
    
    # Specific interesting characters
    chars_to_check = ["˨", "˩", "˦", "ʈ", "ʂ", "ɕ", "ɑ", "ŋ"]
    
    print(f"String: {mandarin_ipa}")
    tokens = tokenizer.encode(mandarin_ipa)
    print(f"Token count: {len(tokens)}")
    print(f"Encoded: {tokens}")
    
    print("\n--- Character Analysis ---")
    for char in chars_to_check:
        token_id = tokenizer.encode(char)
        print(f"'{char}': Encoded as {token_id} (Length: {len(token_id)})")
        if len(token_id) > 1:
            print(f"   -> WARNING: '{char}' is split into multiple tokens! Model usually struggles to learn these if rare.")
        else:
            print(f"   -> OK: Single token.")

if __name__ == "__main__":
    check_tokens()
