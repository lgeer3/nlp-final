import os
import json
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from datasets import load_dataset

FULL_TOKENIZER_PATH = "./tokenizer_custom/tokenizer.json"
SAVE_DIR = "./tokenizer_custom/"
TARGET_VOCAB_SIZE = 20000  # or 20000, 30000...
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]

print(" Loading original tokenizer...")
os.makedirs(SAVE_DIR, exist_ok=True)
tokenizer = Tokenizer.from_file(FULL_TOKENIZER_PATH)
vocab = tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}
tokenizer.model.save("./tokenizer_merges_backup")

with open("./tokenizer_merges_backup/vocab.json", "r", encoding="utf-8") as f:
    original_vocab = json.load(f)


# Load dataset and flatten all text
print(" Loading dataset...")
dataset = load_dataset("dogtooth/default_project_dev_test")
texts = [x["text"] for split in dataset for x in dataset[split] if x["text"].strip()]

# Count token frequency
print(" Scoring token usage...")
token_freq = Counter()
for text in texts:
    encoding = tokenizer.encode(text)
    token_freq.update(encoding.ids)

print(f" Selecting top {TARGET_VOCAB_SIZE} tokens...")
most_common_ids = [tok_id for tok_id, _ in token_freq.most_common(TARGET_VOCAB_SIZE)]
most_common_tokens = {id_to_token[i] for i in most_common_ids if i in id_to_token}
for tok in SPECIAL_TOKENS:
    most_common_tokens.add(tok)

# Ensure all tokens used in merges are preserved
merges_path = "./tokenizer_merges_backup/merges.txt"
with open(merges_path, "r", encoding="utf-8") as f:
    merges = [tuple(line.strip().split()) for line in f if not line.startswith("#")]

tokens_in_merges = set(t for pair in merges for t in pair)
most_common_tokens.update(tokens_in_merges)

# Trim merge list to only valid pairs and track required tokens
valid_merges = []
required_tokens = set(most_common_tokens)

for a, b in merges:
    if a in required_tokens and b in required_tokens:
        merged = a + b
        valid_merges.append((a, b))
        required_tokens.add(merged)

print(f"\n Final token count after merge pruning: {len(required_tokens)}")
print(f" Final merge count: {len(valid_merges)}")

# Build filtered vocab with dense remapped IDs
filtered_vocab = {tok: idx for idx, tok in enumerate(sorted(required_tokens))}

print(f"\n Rebuilding tokenizer with {len(filtered_vocab)} tokens and {len(valid_merges)} merges")

new_model = BPE(vocab=filtered_vocab, merges=valid_merges, unk_token="<unk>")
trimmed_tokenizer = Tokenizer(new_model)
trimmed_tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
trimmed_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
trimmed_tokenizer.decoder = ByteLevelDecoder()

# Save the trimmed tokenizer
trimmed_tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))
print(f"\nTrimmed tokenizer saved to {SAVE_DIR}/tokenizer.json")