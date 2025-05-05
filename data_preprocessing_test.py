import torch
from data_preprocessing.data_preprocessing import preprocess_data
from transformers import AutoTokenizer

def run_preprocessing_tests():
    print("Running preprocessing tests...\n")

    # Minimal test config (adjust if needed)
    dataset = "wikitext"  # Or another HF dataset
    model = "bert-base-cased"
    batch_size = 4
    sequence_length = 32
    vocab_size = 10000
    vocab_trimming = False  # Test both True and False if needed

    train_loader, val_loader, tokenizer, token2id = preprocess_data(
        dataset=dataset,
        batch_size=batch_size,
        vocab_trimming=vocab_trimming,
        vocab_size=vocab_size,
        model=model,
        sequence_length=sequence_length
    )

    print("✅ Tokenizer loaded:", tokenizer.name_or_path)

    def test_dataloader(dataloader, vocab_size, sequence_length, vocab_trimming, split_name="train"):
        first_batch = next(iter(dataloader))
        
        if isinstance(first_batch, (list, tuple)):
            input_ids = first_batch[0]
            attention_mask = first_batch[1] if len(first_batch) > 1 else None
        elif isinstance(first_batch, dict):
            input_ids = first_batch["input_ids"]
            attention_mask = first_batch.get("attention_mask", None)
        else:
            raise ValueError("Unknown batch format")

        # Check input shape
        assert input_ids.ndim == 2, f"{split_name}: Expected input shape (batch, seq_len), got {input_ids.shape}"
        assert input_ids.shape[1] == sequence_length, f"{split_name}: Expected sequence length {sequence_length}, got {input_ids.shape[1]}"

        # Check vocab trimming
        if vocab_trimming:
            max_id = input_ids.max().item()
            assert max_id < vocab_size, f"{split_name}: Token ID {max_id} exceeds vocab_size {vocab_size}"

        # Check attention mask shape
        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape, f"{split_name}: attention_mask shape mismatch"
        
        print(f"{split_name} dataloader looks good ✔")


    test_dataloader(train_loader, vocab_size, sequence_length, vocab_trimming, "Train")
    test_dataloader(val_loader, vocab_size, sequence_length, vocab_trimming, "Validation")


    if vocab_trimming:
        assert token2id is not None, "❌ token2id missing despite vocab_trimming=True"
        print("✅ token2id exists with size:", len(token2id))
    else:
        assert token2id is None, "❌ token2id should be None when vocab_trimming=False"
        print("✅ token2id correctly not returned")

    print("\n🎉 All preprocessing checks completed.\n")

if __name__ == "__main__":
    run_preprocessing_tests()
