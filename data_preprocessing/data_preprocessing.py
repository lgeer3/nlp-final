from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from typing import List, Optional
import json
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

'''
def load_corpus(dataset_name="wikitext", subset="wikitext-103-raw-v1", split="train", sample_size=None) -> List[str]:
    # retrieves the dataset and separates each example by whitescape
    print("about to load corpus")
    data = load_dataset(dataset_name, subset, split=split)
    print("loaded corpus")
    corpus = [x for x in data["text"] if x.strip()]
    

    return corpus
'''
def trim_vocab(vocab: dict, vocab_size: int) -> set:
    # sorts the vocab by the positional score system and trims it to the requested size
    top_tokens = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    return set(tok for tok, _ in top_tokens)

def score_vocab(vocab: dict, tokenizer, corpus: List[str]) -> dict:
    freq = defaultdict(int)
    position = defaultdict(int)
    length = defaultdict(int)

    # counts the per token frequency, occurency
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


def tokenize(text: str,
             tokenizer,
             trimmed_token_set: Optional[set] = None) -> List[int]:
    unk_token = tokenizer.unk_token or "<unk>"
    tokens = tokenizer.tokenize(text)
    if trimmed_token_set:
        tokens = [t if t in trimmed_token_set else unk_token for t in tokens]
    return tokenizer.convert_tokens_to_ids(tokens)
    
def preprocess_data(
    dataset: str,
    batch_size: int,
    vocab_trimming: bool = False,
    vocab_size: int = 10000,
    model: str="bert-base-cased",
    sequence_length: int=64
):
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    print("loaded")

    # Load dataset with proper splits
    print("about to load corpus")
    data = load_dataset(dataset)
    train_texts = [x for x in data["dev"]["text"] if x.strip()]
    val_texts = [x for x in data["dev_test"]["text"] if x.strip()]
    print("loaded corpus")

    sample_size = 5000  # or smaller for debugging

    train_texts = train_texts[:sample_size]
    val_texts = val_texts[:sample_size]

    trimmed_token_set = None
    if vocab_trimming:
        vocab = tokenizer.get_vocab()
        scores = score_vocab(vocab, tokenizer, train_texts)
        trimmed_token_set = trim_vocab(scores, vocab_size)

    def preprocess(texts):
        input_ids = []
        for line in texts:
            ids = tokenize(line, tokenizer, trimmed_token_set)
            input_ids.extend(ids + [tokenizer.sep_token_id])
        
        print(f"Total input_ids length: {len(input_ids)}", flush=True)

        x_data, y_data = [], []
        for i in range(0, len(input_ids) - sequence_length):
            x = input_ids[i:i + sequence_length]
            y = input_ids[i + 1:i + 1 + sequence_length]
            if len(x) == sequence_length and len(y) == sequence_length:
                x_data.append(x)
                y_data.append(y)
        
        print(f"Total x/y pairs: {len(x_data)}", flush=True)

        # Convert to tensors
        x_tensor = torch.tensor(x_data, dtype=torch.long)
        y_tensor = torch.tensor(y_data, dtype=torch.long)

        # Create TensorDataset
        return TensorDataset(x_tensor, y_tensor)

    train_dataset = preprocess(train_texts)
    val_dataset = preprocess(val_texts)

    # Wrap the data into DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader