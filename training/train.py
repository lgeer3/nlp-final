import torch
import time
from tqdm import tqdm
from transformers import get_scheduler
import evaluate

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
            predictions = output['logits']
            predictions = torch.argmax(predictions, dim=1)

            dev_accuracy.add_batch(predictions=predictions, references=labels)

    return dev_accuracy.compute()

def train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=3,
        learning_rate=1e-5,
        gradient_accumulation=8,
        mixed_precision=False,
        save_model=False,
        save_path="./checkpoints/"
    ):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_loader) * epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    best_val_loss = float('inf')
    total_steps = len(train_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader)):
            # If batch is a tuple (input_ids, attention_mask)
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
            # If batch is a dict
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
            
            labels = input_ids.clone()  # For LM, labels = input_ids (shift handled in model)
            
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                output = model(idx=input_ids, targets=labels, mask=attention_mask)
                loss = output['loss'] / gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                total_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == total_steps:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{total_steps}] "
                      f"Loss: {total_loss / (step+1):.4f} "
                      f"Elapsed: {elapsed:.2f}s")


        if total_loss > 0:
            optimizer.step()
            optimizer.zero_grad()


        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                else:  # if batch is a dict
                    input_ids = batch['input_ids'].to(device)
                labels = input_ids.clone()
                output = model(idx=input_ids, targets=labels)
                val_loss += output['loss'].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if save_model and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_path}/best_model_epoch{epoch+1}.pt")
            print(f"Saved best model (loss={avg_val_loss:.4f})")