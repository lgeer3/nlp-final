import torch
from tqdm import tqdm
from transformers import get_scheduler
import evaluate
import sys
import matplotlib.pyplot as plt
import math
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM
import json

def save_perplexity_plot(train_losses, val_losses=None, save_path="perplexity_vs_epochs.png"):
    epochs = range(1, len(train_losses) + 1)
    train_perplexities = [math.exp(loss) for loss in train_losses]
    save_name = f"{save_path}/perplexity_vs_epochs.png"


    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_perplexities, label="Train Perplexity", marker='o')

    val_perplexities = [math.exp(loss) for loss in val_losses]
    plt.plot(epochs, val_perplexities, label="Validation Perplexity", marker='o')

    plt.title("Perplexity vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    print(f" Saved perplexity plot to {save_name}")


def evaluate_model(model, dataloader, device):
    """
    Evaluate a PyTorch Model
    """
    dev_accuracy = evaluate.load('accuracy')
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits
            predictions = torch.argmax(predictions, dim=1)

            dev_accuracy.add_batch(predictions=predictions, references=labels)

    return dev_accuracy.compute()

def train_model(
        model,
        train_loader,
        val_loader,
        device,
        tokenizer,
        epochs=3,
        learning_rate=1e-5,
        gradient_accumulation=8,
        beta=0.5,
        mixed_precision=False,
        knowledge_distill=False,
        save_model=False,
        save_path="./checkpoints/"
    ):
    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_loader) * epochs
    )
    scaler = GradScaler(enabled=mixed_precision)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True, file=sys.stdout)
        for step, batch in enumerate(progress_bar):
            # If batch is a tuple (input_ids, attention_mask)
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
            # If batch is a dict
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            
            labels = input_ids.clone()  # For LM, labels = input_ids (shift handled in model)
            token_count = attention_mask.sum() if attention_mask is not None else input_ids.numel()

            with autocast(enabled=mixed_precision):
                if(knowledge_distill):
                    output = model(idx=input_ids, targets=labels, mask=attention_mask)
                    logits = output.logits

                    with torch.no_grad():
                        teacher_logits = teacher_model(input_ids).logits
                
                    min_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
                    logits = logits[..., :min_vocab_size]
                    teacher_logits = teacher_logits[..., :min_vocab_size]

                    temperature = 2.0
                    softmax = torch.nn.functional.softmax(logits / temperature, dim=-1)
                    log_probs = torch.nn.functional.log_softmax(logits / temperature, dim=-1)

                    kl_loss = torch.nn.functional.kl_div(log_probs, softmax, reduction='batchmean') * (temperature ** 2)

                    ce_loss = output.loss

                    loss = ((1 - beta) * kl_loss + beta * ce_loss) * token_count / gradient_accumulation
                else:
                    output = model(idx=input_ids, targets=labels, mask=attention_mask)
                    assert output.loss is not None, "Loss is None — make sure targets are passed and loss is computed"
                    loss = output.loss / gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
            
            total_loss += (output.loss.item() * token_count)
            total_tokens += token_count

            avg_loss_so_far = total_loss / total_tokens
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")




        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device) if len(batch) > 1 else None
                else:  # if batch is a dict
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                labels = input_ids.clone()
                output = model(idx=input_ids, targets=labels, mask=attention_mask)
                token_count = attention_mask.sum() if attention_mask is not None else input_ids.numel()
                val_loss += output.loss.item() * token_count
                val_tokens += token_count


        avg_val_loss = val_loss / val_tokens
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if save_model and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = f"{save_path}/best_model_epoch{epoch+1}"

            model.save_pretrained(save_dir)
            model.config.save_pretrained(save_dir)

            print(f"Saved best model (loss={avg_val_loss:.4f})")

        prompt = "In the early 20th century, "
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.7)

        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"\n Sample output after epoch {epoch+1}:\n{generated_text}\n")

        
    best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_val_loss = val_losses[best_epoch]

    config_summary = {
        "activation": model.config.activation,
        "norm_type": model.config.norm_type,
        "knowledge_distill": knowledge_distill,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(best_val_loss),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    with open(f"{save_path}/experiment_summary.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    save_perplexity_plot(train_losses, val_losses, save_path=save_path)