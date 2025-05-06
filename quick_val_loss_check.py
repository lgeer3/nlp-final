import torch
import torch.nn.functional as F
from model.model import Model

def test_validation_loss():
    # Step 1: Dummy input (batch=1, seq_len=5)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Step 2: Minimal model
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

    # Step 3: Forward pass
    with torch.no_grad():
        output = model(idx=input_ids, targets=labels, mask=attention_mask)
        loss_model = output.loss.item()
        logits = output.logits

    
    # Step 4: Manual loss calculation
    # Shift both logits and labels
    shifted_logits = logits[:, :-1, :].contiguous()  # [1, 4, vocab]
    shifted_targets = labels[:, 1:].contiguous()     # [1, 4]

    print(shifted_logits.shape)
    print(shifted_targets.shape)

    print("logits shape:", shifted_logits.shape)
    print("targets shape:", shifted_targets.shape)

    # Flatten
    logits_flat = shifted_logits.view(-1, vocab_size)    # [4, vocab]
    targets_flat = shifted_targets.view(-1)              # [4]

    print(logits_flat.shape)
    print(targets_flat.shape)

    loss_manual = F.cross_entropy(logits_flat, targets_flat, reduction='mean').item()

    # Compare
    print(f"🔍 Model loss:  {loss_model:.6f}")
    print(f"🧮 Manual loss: {loss_manual:.6f}")

    assert abs(loss_model - loss_manual) < 1e-5, "❌ Validation loss mismatch!"
    print("✅ Validation loss test passed.")

if __name__ == "__main__":
    test_validation_loss()
