"""
REFERENCE SOLUTION - Memory-Optimized Training Script

This is the solution that demonstrates ALL the memory optimizations.
This is what the agent should produce (or something similar).

Optimizations applied:
1. Mixed Precision (AMP) - autocast + GradScaler
2. Gradient Checkpointing - recompute activations during backward
3. SDPA (FlashAttention) - memory-efficient attention
4. Gradient Accumulation - simulate larger batches
5. Memory-efficient optimizer - set_to_none=True
6. Explicit memory management - empty_cache, del intermediates

DO NOT SHOW THIS TO THE AGENT - this is for grading reference only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import argparse
import time
import math


class TransformerBlock(nn.Module):
    """Single transformer block with memory-efficient attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # Use separate projections for SDPA compatibility
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # OPTIMIZATION 3: Use scaled_dot_product_attention (FlashAttention)
        # This is O(n) memory instead of O(n^2) for the attention matrix
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,  # Efficient causal masking
            dropout_p=0.0 if not self.training else 0.1,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class DeepTransformer(nn.Module):
    """Deep transformer with gradient checkpointing support."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 24,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_checkpointing: bool = True,  # OPTIMIZATION 2: Gradient checkpointing
    ):
        super().__init__()

        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # OPTIMIZATION 2: Gradient checkpointing
                # Recompute activations during backward pass instead of storing them
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.output(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss

        return result


def get_model_config(size: str = "default"):
    """Get model configuration by size."""
    configs = {
        "small": {
            "vocab_size": 10000,
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 8,
            "d_ff": 1024,
            "max_seq_len": 256,
        },
        "default": {
            "vocab_size": 32000,
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 24,
            "d_ff": 3072,
            "max_seq_len": 512,
        },
        "large": {
            "vocab_size": 32000,
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 32,
            "d_ff": 4096,
            "max_seq_len": 1024,
        },
    }
    return configs.get(size, configs["default"])


class RandomTextDataset(Dataset):
    """Synthetic dataset for testing."""

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
    accumulation_steps: int = 4,  # OPTIMIZATION 4: Gradient accumulation
):
    """
    Memory-optimized training loop with multiple optimization techniques.

    Applied optimizations:
    1. Mixed Precision (AMP) - Uses float16 for forward/backward, float32 for optimizer
    2. Gradient Checkpointing - Recomputes activations instead of storing them
    3. SDPA (FlashAttention) - O(n) memory attention instead of O(n^2)
    4. Gradient Accumulation - Simulates larger batch with smaller micro-batches
    5. set_to_none=True - More efficient gradient zeroing
    6. Explicit memory management - torch.cuda.empty_cache() at strategic points
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model with gradient checkpointing enabled
    config = get_model_config(model_size)
    model = DeepTransformer(**config, use_checkpointing=True)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Model config: {config}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # OPTIMIZATION 1: Mixed Precision - GradScaler for loss scaling
    scaler = GradScaler()

    # Create dataset and dataloader
    # Use smaller micro-batches for gradient accumulation
    micro_batch_size = batch_size // accumulation_steps
    dataset = RandomTextDataset(
        vocab_size=config["vocab_size"],
        seq_len=config["max_seq_len"],
        num_samples=num_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        pin_memory=True,  # OPTIMIZATION 6: Faster CPU->GPU transfer
    )

    # Track memory usage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()  # OPTIMIZATION 6: Clear memory before training
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.1f} MB")

    # Training loop
    model.train()
    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)  # OPTIMIZATION 5: More efficient than zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # OPTIMIZATION 1: Mixed Precision - autocast for forward pass
            with autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            # OPTIMIZATION 1: Scaled backward pass
            scaler.scale(loss).backward()

            # OPTIMIZATION 4: Gradient accumulation - only step every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # OPTIMIZATION 5

            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1

            if batch_idx % (10 * accumulation_steps) == 0:
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}, "
                          f"Memory: {current_memory:.1f}MB, Peak: {peak_memory:.1f}MB")
                else:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}")

            # OPTIMIZATION 6: Explicit memory cleanup for long sequences
            del outputs, loss
            if batch_idx % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        total_loss += epoch_loss

    # Report final memory stats
    if device.type == "cuda":
        final_peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n=== Memory Statistics ===")
        print(f"Peak GPU Memory: {final_peak_memory:.1f} MB")
        print("\nOptimizations Applied:")
        print("  1. Mixed Precision (AMP) - autocast + GradScaler")
        print("  2. Gradient Checkpointing - recompute activations")
        print("  3. SDPA (FlashAttention) - O(n) memory attention")
        print("  4. Gradient Accumulation - smaller micro-batches")
        print("  5. set_to_none=True - efficient gradient zeroing")
        print("  6. Explicit memory management - empty_cache, del, pin_memory")

    final_avg_loss = total_loss / num_batches
    print(f"Training complete. Final average loss: {final_avg_loss:.4f}")

    return final_avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized Training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-size", type=str, default="default",
                        choices=["small", "default", "large"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--accumulation-steps", type=int, default=4)

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_size=args.model_size,
        num_samples=args.num_samples,
        accumulation_steps=args.accumulation_steps,
    )
