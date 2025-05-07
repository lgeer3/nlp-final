import json
from tokenizers import Tokenizer
import os

# --- Config ---
INPUT_JSON_PATH = "./tokenizer_custom/tokenizer.json"
VOCAB_KEEP_ITEMS = 20000
SAVE_DIR = f"./tokenizer_trimmed_{VOCAB_KEEP_ITEMS}"

# --- Load tokenizer.json ---
print("\n Loading tokenizer from", INPUT_JSON_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    tokenizer_json = json.load(f)

model = tokenizer_json["model"]
model_type = model["type"]

print(f"🔍 Model type: {model_type}")

if model_type == "BPE":
    vocab = model["vocab"]
    merges = model["merges"]

    print(f"  Trimming vocab to top {VOCAB_KEEP_ITEMS} tokens by ID...")
    new_vocab = {token: i for token, i in vocab.items() if i < VOCAB_KEEP_ITEMS}

    new_merges = []
    for pair in merges:
        if isinstance(pair, str):
            a, b = pair.split()
        else:
            a, b = pair 
        new_token = a + b
        if a in new_vocab and b in new_vocab and new_token in new_vocab:
            new_merges.append(f"{a} {b}")

    print(f" Kept {len(new_vocab)} tokens and {len(new_merges)} merges")

    model["vocab"] = new_vocab
    model["merges"] = new_merges

else:
    raise ValueError(f"Unsupported model type: {model_type}")

# --- Save the new tokenizer JSON ---
from tokenizers import Tokenizer as RawTokenizer
print(f"\n Saving trimmed tokenizer to {SAVE_DIR}/tokenizer.json")
trimmed = RawTokenizer.from_str(json.dumps(tokenizer_json))
trimmed.save(f"{SAVE_DIR}/tokenizer.json")
print("Done.")