import torch
from model.model import Model
from data_preprocessing.prepare_shakespeare_dataset import prepare_shakespeare_data


def quick_val_loss_check():
    # CPU mode only
    device = "cpu"

    # Minimal setup
    batch_size = 2
    sequence_length = 64
    hidden_dim = 128
    hidden_layers = 2
    n_head = 4
    block_size = 64

    print("🔢 Loading small validation batch...")
    _, val_loader, tokenizer, token2id = prepare_shakespeare_data(
        batch_size=batch_size,
        vocab_trimming=False,
        model="bert-base-cased",
        sequence_length=sequence_length
    )

    vocab_size = len(token2id) if token2id else tokenizer.vocab_size

    print("🧠 Initializing small model on CPU...")
    model = Model(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        block_size=block_size,
        n_head=n_head,
        vocab_size=vocab_size,
        token2id=token2id,
    ).to(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    print("🔍 Running single validation step...")
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = input_ids.clone()

            output = model(idx=input_ids, targets=labels, mask=attention_mask)
            loss_tensor = output['loss']

            print(f"Step {step+1}")
            print(" input_ids:", input_ids[0][:10].tolist())
            print(" logits shape:", output['logits'].shape)
            print(" loss:", loss_tensor.item())

            if torch.isnan(loss_tensor):
                print("❌ Loss is NaN — something is wrong!")
            break  # Just one batch

    print("✅ Done.")


if __name__ == "__main__":
    quick_val_loss_check()
