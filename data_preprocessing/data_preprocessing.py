from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from typing import List, Optional, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

def trim_vocab(vocab: dict, vocab_size: int) -> List[str]:
    filtered = [
        (tok, score) for tok, score in vocab.items()
        if not tok.startswith("[unused") and tok not in {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}
    ]
    top_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)[:vocab_size]
    return [tok for tok, _ in top_tokens]

def score_vocab(vocab: dict, tokenizer, corpus: List[str]) -> dict:
    freq = defaultdict(int)
    position = defaultdict(int)
    length = defaultdict(int)

    for line in corpus:
        words = line.strip().split()
        for word in words:
            tokens = tokenizer.tokenize(word)
            for i, token in enumerate(tokens):
                freq[token] += 1
                position[token] += i
                length[token] += len(tokens)

    scores = {}
    for token in freq:
        if freq[token] == 0 or length[token] == 0:
            continue
        avg_position = position[token] / freq[token]
        avg_length = length[token] / freq[token]
        if avg_length == 0:
            continue
        position_score = (avg_length - avg_position) / avg_length
        scores[token] = freq[token] * position_score

    return scores

def tokenize(text, tokenizer, token2id, unk_id):
    tokens = tokenizer.tokenize(text)
    tokens = [t if t in token2id else "<unk>" for t in tokens]
    return [token2id.get(t, unk_id) for t in tokens]

def preprocess_data(
    dataset: str,
    batch_size: int,
    vocab_trimming: bool = False,
    vocab_size: int = 10000,
    model: str = "bert-base-cased",
    sequence_length: int = 64
) -> Tuple[DataLoader, DataLoader, object]:
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    print("loaded tokenizer")

    print("loading dataset...")
    data = load_dataset(dataset)
    train_texts = [x for x in data["dev"]["text"] if x.strip()]
    val_texts = [x for x in data["dev_test"]["text"] if x.strip()]
    print("loaded dataset")

    token2id, unk_id = None, None

    if vocab_trimming:
        print("scoring and trimming vocab...")
        vocab = tokenizer.get_vocab()
        scores = score_vocab(vocab, tokenizer, train_texts)
        trimmed_token_list = trim_vocab(scores, vocab_size)
        token2id = {tok: i for i, tok in enumerate(trimmed_token_list)}
        unk_token = "<unk>"  # Use a clean, custom token
        # Only add it if it doesn't already exist
        if unk_token not in token2id:
            token2id[unk_token] = len(token2id)
        unk_id = token2id[unk_token]
        sep_token = "<sep>"
        if sep_token not in token2id:
            token2id[sep_token] = len(token2id)
        sep_id = token2id[sep_token]
        print(f"trimmed vocab size: {len(token2id)}")

    def preprocess(texts):
        input_ids = []
        for line in texts:
            ids = tokenize(line, tokenizer, token2id=token2id, unk_id=unk_id)
            input_ids.extend(ids + [sep_id])  

        print(f"Total input_ids length: {len(input_ids)}", flush=True)

        MAX_TOKENS = 200_000
        if len(input_ids) > MAX_TOKENS:
            print(f" Truncating input_ids from {len(input_ids)} → {MAX_TOKENS}", flush=True)
            input_ids = input_ids[:MAX_TOKENS]

        x_data, y_data = [], []
        for i in range(0, len(input_ids) - sequence_length):
            x = input_ids[i:i + sequence_length]
            y = input_ids[i + 1:i + 1 + sequence_length]
            if len(x) == sequence_length and len(y) == sequence_length:
                x_data.append(x)
                y_data.append(y)

        print(f"Total training pairs: {len(x_data)}", flush=True)

        x_tensor = torch.tensor(x_data, dtype=torch.long)
        y_tensor = torch.tensor(y_data, dtype=torch.long)

        return TensorDataset(x_tensor, y_tensor)

    train_dataset = preprocess(train_texts)
    val_dataset = preprocess(val_texts)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, tokenizer, token2id
