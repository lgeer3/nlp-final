import argparse
import torch
from data_preprocessing.data_preprocessing import preprocess_data
from model.model import Model
from training.train import train_model



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--block_size', type=int, default=1024, help="Block size for training")
    parser.add_argument('--n_head', type=int, default=1024, help="Block size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument('--gradient_accumulation', type=int, default=8, help="how many steps you accumulate to form a 'large batch'.")
    # Model Architecture
    parser.add_argument('--vocab_trimming', action='store_true', help="Use vocab trimming")
    parser.add_argument('--vocab_size', type=int, default=10000, help="Vocab Size after trimming, only applies if vocab trimming is turned on")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension size for model layers")
    parser.add_argument('--hidden_layers', type=int, default=4, help="Number of hidden layers in the model")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--save_model', action='store_true', help="Save the best model during training")
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help="Where to save model checkpoints")
    return parser.parse_args()

def main():
    print("✅ Python started", flush=True)
    args = parse_args()

    # set_seed(args.seed)
    #set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("PREPROCESSING DATA")

    train_loader, val_loader, tokenizer, token2id = preprocess_data(
        dataset=args.dataset,
        batch_size=args.batch_size,
        vocab_trimming=args.vocab_trimming,
        vocab_size=args.vocab_size,
    )

    print("MAKING MODEL")

    model = Model(
        hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, vocab_size=tokenizer.vocab_size if not args.vocab_trimming else len(token2id), 
                 block_size=args.block_size, n_head=args.n_head, attn_pdrop=0.1, resid_pdrop=0.1, 
                 embd_pdrop=0.1, token2id=token2id
    )
    model = model.to(device)

    print("TRAINING MODEL")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_model=args.save_model,
        save_path=args.save_path,
        gradient_accumulation=args.gradient_accumulation,
        tokenizer=tokenizer
    )




if __name__ == "__main__":
    main()
