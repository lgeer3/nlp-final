import torch
import torch.nn.functional as F
from model.model import Model

def test_validation_loss():
    # Step 1: Dummy input (batch=1, seq_len=5)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])  # shape: [1, 5]
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Step 2: Minimal model config
    vocab_size = 100
    hidden_dim = 32
    block_size = 5
    model = Model(
        hidden_dim=hidden_dim,
        hidden_layers=1,
        block_size=block_size,
        n_head=2,
        vocab_size=vocab_size,
    )

    model.eval()

    # Step 3: Forward pass (no gradients)
    with torch.no_grad():
        output = model(idx=input_ids, targets=labels, mask=attention_mask)
        loss_model = output['loss'].item()
        logits = output['logits']

    # Step 4: Shift targets to match model’s slicing
    # Model already did: logits = logits[:, :-1, :]
    targets = labels[:, 1:]  # shape: [1, 4]
    logits = logits          # already shape: [1, 4, vocab_size]

    print("targets shape:", targets.shape)
    print("logits shape:", logits.shape)

    # Step 5: Flatten for loss
    logits_flat = logits.view(-1, vocab_size)   # [4, vocab]
    targets_flat = targets.view(-1)             # [4]

    print("logits_flat:", logits_flat.shape)
    print("targets_flat:", targets_flat.shape)

    # Step 6: Manual loss calculation
    loss_manual = F.cross_entropy(logits_flat, targets_flat, reduction='mean').item()

    # Step 7: Compare
    print(f"\n🔍 Model loss:  {loss_model:.6f}")
    print(f"🧮 Manual loss: {loss_manual:.6f}")

    assert abs(loss_model - loss_manual) < 1e-5, "❌ Validation loss mismatch!"
    print("✅ Validation loss test passed.")

if __name__ == "__main__":
    test_validation_loss()
