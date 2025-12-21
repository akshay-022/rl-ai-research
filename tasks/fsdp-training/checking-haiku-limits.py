"""
Test Haiku's ability to convert single-GPU training to FSDP.

This script:
1. Sends the single-GPU code to Haiku and asks it to convert to FSDP
2. Sends Haiku's output to Sonnet for evaluation
3. Prints the results
"""

import os
from dotenv import load_dotenv

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import anthropic

SINGLE_GPU_CODE = '''
"""
Single-GPU Training Script - Convert this to FSDP
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

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

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
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--save-path", type=str, default="checkpoint.pt")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_size=args.model_size,
        save_path=args.save_path,
    )
'''

HAIKU_PROMPT = """Convert this single-GPU PyTorch training script to use FSDP (Fully Sharded Data Parallel) for distributed training.

Requirements:
1. Use torch.distributed and FSDP from torch.distributed.fsdp
2. Use DistributedSampler instead of shuffle=True
3. Use transformer_auto_wrap_policy to wrap TransformerBlock layers
4. Handle LOCAL_RANK environment variable
5. Call sampler.set_epoch(epoch) in the training loop
6. Save checkpoint only on rank 0 with proper FSDP state dict handling
7. Initialize and destroy the process group properly
8. The script should be runnable with: torchrun --nproc_per_node=2 script.py

Here's the code to convert:

```python
{code}
```

Return ONLY the converted Python code, no explanations."""

SONNET_EVAL_PROMPT = """You are evaluating an FSDP implementation. Check if this code correctly implements FSDP distributed training.

Grade each of these 10 requirements (1 point each):
1. dist.init_process_group() is called
2. FSDP is imported from torch.distributed.fsdp
3. Model is wrapped with FSDP()
4. DistributedSampler is used
5. transformer_auto_wrap_policy is used with TransformerBlock
6. sampler.set_epoch(epoch) is called in training loop
7. Checkpoint saving checks rank == 0
8. FSDP state_dict_type context is used for saving
9. dist.destroy_process_group() is called
10. LOCAL_RANK environment variable is read

Code to evaluate:
```python
{code}
```

For each requirement, say PASS or FAIL with a brief reason.
Then give a final score out of 10.
Finally, state if this would be considered a PASSING implementation (8+ points)."""


def main():
    client = anthropic.Anthropic()

    print("=" * 60)
    print("STEP 1: Asking Haiku to convert to FSDP")
    print("=" * 60)

    haiku_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": HAIKU_PROMPT.format(code=SINGLE_GPU_CODE)
        }]
    )

    haiku_code = haiku_response.content[0].text

    # Extract code if wrapped in markdown
    if "```python" in haiku_code:
        haiku_code = haiku_code.split("```python")[1].split("```")[0]
    elif "```" in haiku_code:
        haiku_code = haiku_code.split("```")[1].split("```")[0]

    print("\n--- Haiku's FSDP Code ---")
    print(haiku_code[:2000] + "..." if len(haiku_code) > 2000 else haiku_code)

    print("\n" + "=" * 60)
    print("STEP 2: Asking Sonnet to evaluate")
    print("=" * 60)

    sonnet_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": SONNET_EVAL_PROMPT.format(code=haiku_code)
        }]
    )

    evaluation = sonnet_response.content[0].text

    print("\n--- Sonnet's Evaluation ---")
    print(evaluation)

    print("\n" + "=" * 60)
    print("FULL HAIKU OUTPUT (for reference)")
    print("=" * 60)
    print(haiku_code)


if __name__ == "__main__":
    main()
