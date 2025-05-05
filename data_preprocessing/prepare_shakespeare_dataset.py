import os
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from data_preprocessing.data_preprocessing import preprocess_data


def download_shakespeare(path="shakespeare/input.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("📥 Downloading TinyShakespeare dataset...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
        print("✅ Downloaded to", path)
    return path


def prepare_shakespeare_data(
    batch_size=64,
    vocab_trimming=False,
    vocab_size=10000,
    model="bert-base-cased",
    sequence_length=128,
):
    # Download dataset
    path = download_shakespeare()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into train/val
    split_idx = int(len(text) * 0.9)
    train_lines = [text[:split_idx]]
    val_lines = [text[split_idx:]]

    # Create HuggingFace-compatible dataset
    dataset_dict = DatasetDict({
        "dev": Dataset.from_dict({"text": train_lines}),
        "dev_test": Dataset.from_dict({"text": val_lines}),
    })

    # Monkey-patch HuggingFace's load_dataset to return our data
    import datasets
    datasets.load_dataset = lambda *args, **kwargs: dataset_dict

    # Use your existing preprocessing pipeline
    return preprocess_data(
        dataset="any_name",  # ignored due to monkey-patch
        batch_size=batch_size,
        vocab_trimming=vocab_trimming,
        vocab_size=vocab_size,
        model=model,
        sequence_length=sequence_length
    )
