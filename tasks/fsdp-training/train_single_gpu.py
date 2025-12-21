"""
Single-GPU Training Script

This is the STARTER CODE that the agent must convert to use FSDP.

The agent should modify this file to support distributed training with
Fully Sharded Data Parallel (FSDP).

DO NOT MODIFY THIS FILE - this is what the agent receives as input.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import argparse
import time

from model import SimpleTransformer, TransformerBlock, get_model_config


class RandomTextDataset(Dataset):
    """Synthetic dataset for testing - generates random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens}


def train(
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    model_size: str = "small",
    save_path: str = "checkpoint.pt",
):
    """
    Single-GPU training loop.

    TODO: Convert this to use FSDP for distributed training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    config = get_model_config(model_size)
    model = SimpleTransformer(**config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = RandomTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=config["max_seq_len"],
        num_samples=1000,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        total_loss += epoch_loss

    # Save checkpoint
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

    final_avg_loss = total_loss / num_batches
    print(f"Training complete. Final average loss: {final_avg_loss:.4f}")

    return final_avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--save-path", type=str, default="checkpoint.pt")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_size=args.model_size,
        save_path=args.save_path,
    )
