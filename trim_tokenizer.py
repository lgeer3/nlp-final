import os
import json
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
from datasets import load_dataset
from tokenizers.trainers import BpeTrainer

FULL_TOKENIZER_PATH = "./tokenizer_custom/tokenizer.json"
SAVE_DIR = "./tokenizer_custom/"
TARGET_VOCAB_SIZE = 20000  # or 20000, 30000...
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]

print("🔹 Loading original tokenizer...")
os.makedirs(SAVE_DIR, exist_ok=True)
original_tokenizer = Tokenizer.from_file(FULL_TOKENIZER_PATH)
vocab = original_tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

# Load dataset and flatten all text
print("🔹 Loading dataset...")
dataset = load_dataset("dogtooth/default_project_dev_test")
texts = [x["text"] for split in dataset for x in dataset[split] if x["text"].strip()]

# Count token frequency
print("🔹 Scoring token usage...")
token_freq = Counter()
for text in texts:
    encoding = original_tokenizer.encode(text)
    token_freq.update(encoding.ids)

print(f"🔹 Selecting top {TARGET_VOCAB_SIZE} tokens...")
most_common_ids = [tok_id for tok_id, _ in token_freq.most_common(TARGET_VOCAB_SIZE)]
most_common_tokens = {id_to_token[i] for i in most_common_ids if i in id_to_token}
for tok in SPECIAL_TOKENS:
    most_common_tokens.add(tok)

print(f"✅ Retained {len(most_common_tokens)} tokens total (with specials)")


# Extract text again so we can re-learn merges from just top tokens
filtered_texts = []
for text in texts:
    tokens = original_tokenizer.encode(text).tokens
    tokens = [t if t in most_common_tokens else "<unk>" for t in tokens]
    filtered_texts.append(" ".join(tokens))


print("🔄 Re-training tokenizer using trimmed vocab...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=len(most_common_tokens),
    special_tokens=list(SPECIAL_TOKENS)
)
tokenizer.train_from_iterator(filtered_texts, trainer=trainer)

# Save copy of tokenizer for consistency
tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))

print(f"📦 Saved to: {SAVE_DIR}")