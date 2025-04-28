import argparse
import torch
from data_preprocessing.data_preprocessing import preprocess_data
from model.model import Model
from training.train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with flexible options.")

    # Core flags
    parser.add_argument('--rmsnorm', action='store_true', help="Use RMSNorm instead of LayerNorm")
    parser.add_argument('--distillation', action='store_true', help="Use knowledge distillation")
    parser.add_argument('--vocab_trimming', action='store_true', help="Use vocab trimming")
    parser.add_argument('--distillation_model', type=str, help="Pretrained model name or path (e.g., 'gpt2' or a local path)")
    parser.add_argument('--activation', type=str, help="Type of activation used for model (eg. GeGLU)")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--block_size', type=int, default=32, help="Block size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument('--gradient_accumulation', type=int, default=8, help="how many steps you accumulate to form a 'large batch'.")

    # Model Architecture
    parser.add_argument('--vocab_size', type=int, default=10000, help="Vocab Size after trimming, only applies if vocab trimming is turned on")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension size for model layers")
    parser.add_argument('--hidden_layers', type=int, default=4, help="Number of hidden layers in the model")
    parser.add_argument('--mixed_precision', action='store_true', help="Use mixed precision training")

    # Extras
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--save_model', action='store_true', help="Save the best model during training")
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help="Where to save model checkpoints")

    return parser.parse_args()

def main():
    args = parse_args()

    # set_seed(args.seed)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = preprocess_data(
        dataset=args.dataset,
        batch_size=args.batch_size,
        vocab_trimming=args.vocab_trimming,
        vocab_size=args.vocab_size,
    )

    model = Model(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        rmsnorm=args.rmsnorm,
        activation=args.activation,
        vocab_size=args.vocab_size,
    )
    model = model.to(device)

    train_model(
        model=model,
        train_loader=train_loader
        val_loader=val_loader
        device=device
        epochs=args.epochs
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation
        use_distillation=args.distillation,
        mixed_precision=args.mixed_precision,
        save_model=args.save_model,
        save_path=args.save_path,
    )



if __name__ == "__main__":
    main()
