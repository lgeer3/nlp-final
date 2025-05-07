from datasets import load_dataset
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import os
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Parameters
dataset_name = "dogtooth/default_project_dev_test"
vocab_size = 30000
save_dir = "./tokenizer_custom/"

print("🔹 Loading dataset...")
dataset = load_dataset(dataset_name)
texts = [x["text"] for x in dataset["dev"] if x["text"].strip()] + \
        [x["text"] for x in dataset["dev_test"] if x["text"].strip()]

print(f"🔹 Training tokenizer on {len(texts)} samples...")

# Initialize tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>", "<sep>"]
)

# Train from iterator
tokenizer.train_from_iterator(texts, trainer=trainer)

# Post-processing to ensure correct padding behavior
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A <sep>",
    pair="$A <sep> $B:1 <sep>:1",
    special_tokens=[("<sep>", tokenizer.token_to_id("<sep>"))],
)

tokenizer.decoder = ByteLevelDecoder()
tokenizer.enable_truncation(max_length=256)

# Save tokenizer files
os.makedirs(save_dir, exist_ok=True)
tokenizer.model.save(save_dir)
tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

print(f"✅ Saved custom tokenizer to {save_dir}")
