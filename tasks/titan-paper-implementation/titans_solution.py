"""
Titans Architecture - REFERENCE SOLUTION (Gold Standard)

This is what a correct Titans implementation should look like.
DO NOT show this to the agent - it's for grading reference only.
"""

import torch
import torch.nn as nn
import math


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory Module (Titans).
    Implements a dynamic MLP where weights serve as the memory state.
    """
    def __init__(self, d_model: int, d_memory: int, dropout: float = 0.1):
        super().__init__()
        self.d_memory = d_memory

        # Projections for Memory Operations
        self.w_key = nn.Linear(d_model, d_memory)
        self.w_value = nn.Linear(d_model, d_memory)
        self.w_query = nn.Linear(d_model, d_memory)
        self.w_out = nn.Linear(d_memory, d_model)

        # --- SURPRISE METRIC ---
        # A learnable gate that determines how much to "write" to memory
        # based on the input's unpredictability/importance.
        self.surprise_gate = nn.Linear(d_model, 1)

        # Learnable Decay (controls how fast memory fades)
        self.decay = nn.Parameter(torch.tensor(0.95))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        b, s, d = x.shape

        # Initialize Memory State (The "Weight Matrix" M)
        if state is None:
            state = torch.zeros(b, self.d_memory, self.d_memory, device=x.device)

        # Pre-compute projections for the whole sequence
        k = self.w_key(x)   # (B, S, D_mem)
        v = self.w_value(x)  # (B, S, D_mem)
        q = self.w_query(x)  # (B, S, D_mem)

        # Calculate Surprise for the whole sequence
        # (B, S, D) -> (B, S, 1)
        surprise = torch.sigmoid(self.surprise_gate(x))

        outputs = []
        curr_state = state

        # --- RECURRENT LOOP ---
        for t in range(s):
            # Shape: (B, D_mem, 1) or (B, 1, D_mem) for matrix multiplication
            k_t = k[:, t].unsqueeze(2)
            v_t = v[:, t].unsqueeze(2)
            q_t = q[:, t].unsqueeze(1)
            s_t = surprise[:, t].unsqueeze(2)  # (B, 1, 1)

            # 1. READ (Retrieval): Y = M * Q
            # We treat the state as a linear layer weights applied to Query
            mem_out = torch.bmm(q_t, curr_state).squeeze(1)
            outputs.append(mem_out)

            # 2. WRITE (Update): M_new = Decay * M + Surprise * (V * K.T)
            # The outer product (V * K.T) creates the associative binding
            association = torch.bmm(v_t, k_t.transpose(1, 2))

            # Weighted update based on surprise
            update_term = s_t * association
            curr_state = self.decay * curr_state + update_term

        # Stack outputs back into sequence
        outputs = torch.stack(outputs, dim=1)  # (B, S, D_mem)

        # Project back to model dimension
        return self.w_out(outputs), curr_state


class TitansBlock(nn.Module):
    """
    Titans Block: Combines Attention (Short-term) and Neural Memory (Long-term).
    Uses 'Memory as a Gate' (MAG) fusion.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_memory: int, dropout: float = 0.1):
        super().__init__()
        # 1. Short-term Memory (Attention)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # 2. Long-term Memory (Neural Memory)
        self.neural_memory = NeuralMemory(d_model, d_memory, dropout)

        # 3. Gating Mechanism
        self.gate = nn.Linear(d_model * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, mem_state: torch.Tensor = None):
        # Run branches in parallel
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        mem_out, new_state = self.neural_memory(x, mem_state)

        # Fuse branches
        combined = torch.cat([attn_out, mem_out], dim=-1)
        g = torch.sigmoid(self.gate(combined))

        # Weighted combination
        fused = g * attn_out + (1 - g) * mem_out

        # Residual + Norm
        x = self.norm1(x + fused)

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, new_state


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        d_memory: int = 128  # Added config for memory size
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Swapped TransformerBlock for TitansBlock
        self.layers = nn.ModuleList([
            TitansBlock(d_model, n_heads, d_ff, d_memory, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.embedding.weight

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
        x = self.embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Standard Causal Mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        # We maintain memory state between layers if needed, but for simple training
        # we reset it (pass None) at the start of every forward pass.
        # Ideally, this should be stateful across batches for true infinite context.
        mem_state = None

        for layer in self.layers:
            # We ignore the state output for now as we don't persist it across batches
            x, _ = layer(x, mask=causal_mask, mem_state=mem_state)

        x = self.norm(x)
        logits = self.output(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result["loss"] = loss

        return result


if __name__ == "__main__":
    # Quick test
    model = SimpleTransformer(vocab_size=1000, d_model=256, n_heads=4, n_layers=2, d_memory=64)
    input_ids = torch.randint(0, 1000, (2, 64))
    output = model(input_ids)
    print(f"Logits shape: {output['logits'].shape}")
    print("Titans model working correctly!")
