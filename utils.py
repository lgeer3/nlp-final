import numpy as np
import random
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {load_path}")
    return model



