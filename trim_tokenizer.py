from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from datasets import load_dataset
from collections import Counter
import os
import json

FULL_TOKENIZER_PATH = "./tokenizer_custom/tokenizer.json"
TRIMMED_TOKENIZER_SAVE_DIR = "./tokenizer_trimmed_{}"
TARGET_VOCAB_SIZE = 20000  # or 20000, 30000...

print("🔹 Loading original tokenizer...")
tokenizer = Tokenizer.from_file(FULL_TOKENIZER_PATH)

# Load dataset and flatten all text
print("🔹 Loading dataset...")
dataset = load_dataset("dogtooth/default_project_dev_test")
texts = [x["text"] for split in dataset for x in dataset[split] if x["text"].strip()]

print("🔹 Encoding dataset...")
counter = Counter()
for text in texts:
    encoding = tokenizer.encode(text)
    for id in encoding.ids:
        counter[id] += 1

# Sort token IDs by frequency
most_common_ids = [tok_id for tok_id, _ in counter.most_common(TARGET_VOCAB_SIZE)]

# Map token IDs to their strings
id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
kept_tokens = [id_to_token[i] for i in most_common_ids if i in id_to_token]

# Add special tokens explicitly
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<sep>"]
for token in special_tokens:
    if token not in kept_tokens:
        kept_tokens.append(token)

print(f"Trimmed vocab: {len(kept_tokens)} tokens (target was {TARGET_VOCAB_SIZE})")

# Rebuild tokenizer with limited vocab
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=len(kept_tokens),
    special_tokens=special_tokens,
    initial_alphabet=[],
    vocab=None
)

tokenizer.train_from_iterator(texts, trainer=trainer)

save_path = TRIMMED_TOKENIZER_SAVE_DIR.format(len(kept_tokens))
os.makedirs(save_path, exist_ok=True)
tokenizer.save(os.path.join(save_path, "tokenizer.json"))
print(f"✅ Trimmed tokenizer saved to {save_path}")
