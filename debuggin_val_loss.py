from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from data_preprocessing.data_preprocessing import preprocess_data
from utils import load_model

device = "cuda"
model = AutoModelForCausalLM()
load_model(model, "checkpoints/best_model_epoch4.pt")

train_loader, val_loader, tokenizer, token2id = preprocess_data(
        dataset='dogtooth/default_project_dev_test',
        batch_size=64,
        vocab_trimming=False,
        vocab_size=50257,
        sequence_length = 256
    )

batch = next(iter(val_loader))
input_ids = batch['input_ids'].to(device)
labels = input_ids.clone()
attention_mask = batch['attention_mask'].to(device)
output = model(idx=input_ids, targets=labels, mask=attention_mask)
manual_loss = F.cross_entropy(output.logits.view(-1, output.logits.size(-1)), labels.view(-1), ignore_index=-100)
print("HF loss:", output.loss.item(), "manual:", manual_loss.item())