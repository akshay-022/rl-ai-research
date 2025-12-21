"""
FSDP Training Script - REFERENCE SOLUTION

This is what a correct FSDP implementation should look like.
DO NOT show this to the agent - it's for grading reference only.

Run with: torchrun --nproc_per_node=2 train_fsdp_solution.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
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
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens}


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train(
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    model_size: str = "small",
    save_path: str = "checkpoint.pt",
):
    """
    FSDP distributed training loop.
    """
    # Initialize distributed
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Starting FSDP training with {world_size} GPUs")
        print(f"Local rank: {local_rank}")

    # Create model (on CPU first, FSDP will move to GPU)
    config = get_model_config(model_size)
    model = SimpleTransformer(**config)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters")

    # Define auto wrap policy - wrap each TransformerBlock
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Optional: Mixed precision for efficiency
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
    )

    # Create optimizer AFTER FSDP wrapping
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=config["max_seq_len"],
        num_samples=1000,
    )

    # Use DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Training loop
    model.train()
    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        # Critical: Set epoch for proper shuffling
        sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            # Only log on rank 0
            if rank == 0 and batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Synchronize loss across ranks
        epoch_loss_tensor = torch.tensor([epoch_loss], device=local_rank)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = epoch_loss_tensor.item() / (len(dataloader) * world_size)

        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")

        total_loss += epoch_loss

    # Save checkpoint only on rank 0 with proper FSDP handling
    if rank == 0:
        print(f"Saving model to {save_path}")

        # Use FSDP state dict context for proper saving
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
            torch.save(state_dict, save_path)

    # Wait for all ranks before cleanup
    dist.barrier()

    final_avg_loss = total_loss / num_batches
    if rank == 0:
        print(f"Training complete. Final average loss: {final_avg_loss:.4f}")

    # Cleanup
    cleanup_distributed()

    return final_avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP Training")
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
