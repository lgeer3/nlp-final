from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from datasets import load_dataset
from collections import Counter
import os
import json

FULL_TOKENIZER_PATH = "./tokenizer_custom/tokenizer.json"
SAVE_DIR = "./tokenizer_custom/"
TARGET_VOCAB_SIZE = 20000  # or 20000, 30000...
os.makedirs(SAVE_DIR, exist_ok=True)

print("🔹 Loading original tokenizer...")
tokenizer = Tokenizer.from_file(FULL_TOKENIZER_PATH)

# Load dataset and flatten all text
print("🔹 Loading dataset...")
dataset = load_dataset("dogtooth/default_project_dev_test")
texts = [x["text"] for split in dataset for x in dataset[split] if x["text"].strip()]

# Count token frequency
print("🔹 Scoring token frequency...")
token_counts = Counter()
for text in texts:
    encoding = tokenizer.encode(text)
    for token_id in encoding.ids:
        token_counts[token_id] += 1

# Sort tokens by frequency
most_common_ids = [tok_id for tok_id, _ in token_counts.most_common(TARGET_VOCAB_SIZE)]


# Preserve special tokens
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]
vocab = tokenizer.get_vocab()
special_ids = [vocab[tok] for tok in special_tokens if tok in vocab]

# Final allowed token set
allowed_token_ids = set(most_common_ids + special_ids)
all_token_ids = set(vocab.values())
banned_token_ids = sorted(list(all_token_ids - allowed_token_ids))

print(f"✅ Keeping {len(allowed_token_ids)} tokens | Banning {len(banned_token_ids)}")

# Save banned token IDs
banned_path = os.path.join(SAVE_DIR, "banned_token_ids.json")
with open(banned_path, "w") as f:
    json.dump(banned_token_ids, f)

# Save copy of tokenizer for consistency
tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))

print(f"📦 Saved to: {SAVE_DIR}")