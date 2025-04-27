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
        use_distillation=False,
        mixed_precision=False,
        save_model=False,
        save_path="./checkpoints/"
    ):
    """
    Train a PyTorch model
    """

    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_loader) * epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_accuracy = evaluate.load('accuracy')

        print(f"\nEpoch {epoch + 1}/{epochs}:")

        optimizer.zero_grad()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output['logits']
                loss = loss_fn(logits, labels) / gradient_accumulation

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

            preds = torch.argmax(logits, dim=1)
            train_accuracy.add_batch(predictions=preds, references=labels)

        train_metrics = train_accuracy.compute()
        print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_acc = val_metrics['accuracy']
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save the model if it's the best so far
        if save_model and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_path}/best_model_epoch{epoch+1}.pt")
            print(f"Saved new best model at epoch {epoch + 1} with accuracy {val_acc:.4f}")

    print("\nTraining completed.")

