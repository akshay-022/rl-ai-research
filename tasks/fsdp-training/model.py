import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


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
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
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

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

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


def get_model_config(size: str = "small"):
    configs = {
        "tiny": {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 512,
            "max_seq_len": 128,
        },
        "small": {
            "vocab_size": 10000,
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "max_seq_len": 512,
        },
        "medium": {
            "vocab_size": 32000,
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 12,
            "d_ff": 4096,
            "max_seq_len": 1024,
        },
    }
    return configs.get(size, configs["small"])
