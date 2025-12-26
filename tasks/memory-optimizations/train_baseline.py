"""
Baseline Training Script - NO Memory Optimizations

This is the STARTER CODE that the agent must optimize.
It trains a deep transformer model with NO memory optimizations applied.

The agent should modify this to use memory optimization techniques to reduce
peak GPU memory usage while maintaining training functionality.

DO NOT MODIFY THIS FILE - this is what the agent receives as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import argparse
import time

from model import DeepTransformer, TransformerBlock, get_model_config


class RandomTextDataset(Dataset):
    """Synthetic dataset for testing - generates random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens}


def train(
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    model_size: str = "default",
    num_samples: int = 500,
):
    """
    Baseline training loop with NO memory optimizations.

    Current issues (that the agent should fix):
    1. Uses float32 for all computations (wastes memory)
    2. Stores all activations for backward pass (high memory)
    3. Uses naive attention implementation (O(n^2) memory for attention matrix)
    4. No memory-efficient gradient accumulation

    Target: Apply memory optimizations to reduce peak GPU memory usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model (no optimizations - full float32)
    config = get_model_config(model_size)
    model = DeepTransformer(**config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Model config: {config}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = RandomTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=config["max_seq_len"],
        num_samples=num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Track memory usage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.1f} MB")

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

            # Forward pass (no mixed precision - wastes memory)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                          f"Memory: {current_memory:.1f}MB, Peak: {peak_memory:.1f}MB")
                else:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        total_loss += epoch_loss

    # Report final memory stats
    if device.type == "cuda":
        final_peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n=== Memory Statistics ===")
        print(f"Peak GPU Memory: {final_peak_memory:.1f} MB")

    final_avg_loss = total_loss / num_batches
    print(f"Training complete. Final average loss: {final_avg_loss:.4f}")

    return final_avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Training (No Optimizations)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-size", type=str, default="default",
                        choices=["small", "default", "large"])
    parser.add_argument("--num-samples", type=int, default=500)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_size=args.model_size,
        num_samples=args.num_samples,
    )
