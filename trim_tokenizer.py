import json
from tokenizers import Tokenizer
import os
from transformers import PreTrainedTokenizerFast

# --- Config ---
INPUT_JSON_PATH = "./model_output/tokenizer.json"
VOCAB_KEEP_ITEMS = 30000
SAVE_DIR = f"./model_output/"

# --- Load tokenizer.json ---
print("\n Loading tokenizer from", INPUT_JSON_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    tokenizer_json = json.load(f)

model = tokenizer_json["model"]


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

from tokenizers import Tokenizer as RawTokenizer

trimmed_tokenizer = RawTokenizer.from_str(json.dumps(tokenizer_json))

print(f"\n Saving trimmed tokenizer to {SAVE_DIR}/tokenizer.json")
trimmed = RawTokenizer.from_str(json.dumps(tokenizer_json))
wrapped = PreTrainedTokenizerFast(tokenizer_file=os.path.join(SAVE_DIR, "tokenizer.json"))
wrapped.pad_token = "<pad>"
wrapped.unk_token = "<unk>"
wrapped.sep_token = "<sep>"

# Save in Hugging Face format
wrapped.save_pretrained(SAVE_DIR)