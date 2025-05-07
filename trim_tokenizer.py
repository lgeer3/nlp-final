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

print(f" Retained {len(most_common_tokens)} tokens total (with specials)")


# Extract text again so we can re-learn merges from just top tokens
model_info = tokenizer._tokenizer.model
merges = model_info.get_merges()
filtered_vocab = {tok: i for i, tok in enumerate(sorted(most_common_tokens))}


print(f" Rebuilding tokenizer with {len(filtered_vocab)} tokens and {len(merges)} merges")

new_model = BPE(vocab=filtered_vocab, merges=merges, unk_token="<unk>")

trimmed_tokenizer = Tokenizer(new_model)
trimmed_tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
trimmed_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
trimmed_tokenizer.decoder = ByteLevelDecoder()

# --- Save the trimmed tokenizer ---
trimmed_tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))
print(f" Trimmed tokenizer saved to {SAVE_DIR}/tokenizer.json")