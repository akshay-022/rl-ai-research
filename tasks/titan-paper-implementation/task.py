"""
Titans Architecture Implementation Task

This file contains:
- PROMPT: The challenge prompt given to the agent (with technical specs).
- TOOLS: The tool definitions available to the agent.
- grading_func: Function that validates the agent's Neural Memory implementation using Claude Sonnet 4.5.
"""

import os
import ast
from typing import Any, TypedDict
from collections.abc import Callable

import anthropic
from dotenv import load_dotenv

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# --- STARTER CODE (The Simple Transformer) ---
STARTER_CODE = """
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
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_ids, labels=None):
        b, s = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(torch.arange(s, device=input_ids.device))
        x = self.dropout(x)
        mask = torch.triu(torch.ones(s, s, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.output(self.norm(x))
        return {"logits": logits}
"""

# --- THE CHALLENGE PROMPT ---
PROMPT = f"""You are a Senior AI Architect. Your task is to implement the **Titans** architecture from the Google Research paper "Titans: Learning to Memorize at Test Time".

## Starter Code
```python
{STARTER_CODE}
```

## Task

Modify this Transformer to implement the Titans architecture with Neural Long-Term Memory. Use the "Memory as a Gate" (MAG) variant from the paper.

Preserve the input/output signature of the model (`input_ids` -> `logits`).

## Submission

Submit your complete, runnable Python code as a string.
"""


# --- TOOLS & GRADING ---

class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    return {"answer": answer, "submitted": True}


SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your complete Titans implementation code",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}

TOOLS = [SUBMIT_TOOL]
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {"submit_answer": submit_answer_tool}


GRADING_RUBRIC = """You are a strict technical evaluator for AI architecture implementations.

Evaluate the submitted code against the Titans Architecture specification. Be STRICT and RIGOROUS.

## Grading Criteria (6 points total, need 6/6 to pass)

### 1. NeuralMemory Class (1 point)
- MUST define a class named `NeuralMemory` (or very similar like `NeuralLongTermMemory`)
- MUST be an nn.Module with proper __init__ and forward methods
- FAIL if: Just a placeholder, missing forward method, or no actual memory logic

### 2. Surprise Gate with Sigmoid (1 point)
- MUST implement a learned "surprise" metric that controls how much to write to memory
- MUST use sigmoid activation (torch.sigmoid, F.sigmoid, or nn.Sigmoid)
- The surprise should be computed from the input and used to gate the memory update
- FAIL if: No surprise mechanism, hardcoded values, or missing sigmoid

### 3. Learnable Decay Parameter (1 point)
- MUST have a learnable decay/eta parameter (nn.Parameter)
- This controls how fast old memory fades
- Should be used in the memory update equation: M_new = decay * M_old + ...
- FAIL if: Hardcoded decay, no decay at all, or decay not used in update

### 4. Outer Product for Memory Association (1 point)
- MUST compute V ⊗ K^T (outer product) to create associative memory
- This is typically: torch.bmm(v.unsqueeze(-1), k.unsqueeze(-2)) or v @ k.transpose(-1,-2)
- The outer product creates the key-value binding stored in memory
- FAIL if: No outer product, just concatenation, or incorrect dimensions

### 5. Recurrent Memory State Update (1 point)
- MUST have a loop (for t in range(seq_len)) that updates memory state step by step
- Each step: read from memory, then write to memory
- Memory state must persist across timesteps within a sequence
- The update rule: M = decay * M + surprise * (V ⊗ K^T)
- FAIL if: No recurrent loop, processing all at once without state updates, or batched without per-step updates

### 6. Memory-Attention Gating/Fusion (1 point)
- MUST combine attention output and memory output using a learned gate
- Gate formula: g = sigmoid(W[attn; mem]) then out = g * attn + (1-g) * mem
- Or similar weighted combination with learned parameters
- FAIL if: No fusion, just addition, or no learned gating mechanism

## Response Format

For each criterion, respond with exactly this format:
```
1. NeuralMemory Class: PASS/FAIL - [brief reason]
2. Surprise Gate: PASS/FAIL - [brief reason]
3. Learnable Decay: PASS/FAIL - [brief reason]
4. Outer Product: PASS/FAIL - [brief reason]
5. Recurrent Update: PASS/FAIL - [brief reason]
6. Memory-Attention Gating: PASS/FAIL - [brief reason]

TOTAL: X/6
RESULT: PASS/FAIL
```

A submission PASSES only if it scores 6/6. Be strict - partial implementations should fail."""


def grading_func(result: Any) -> bool:
    """
    Validates the Titans implementation using Claude Sonnet 4.5 as evaluator.
    Requires 6/6 criteria to pass.
    """
    if not result or not isinstance(result, str):
        print("FAIL: No code submitted or not a string")
        return False

    code = result.strip()

    # Remove markdown code blocks if present
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    # Check valid Python syntax first
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"FAIL: Syntax error in submitted code: {e}")
        return False

    # Use Claude Sonnet 4.5 to evaluate
    client = anthropic.Anthropic()

    eval_prompt = f"""{GRADING_RUBRIC}

## Code to Evaluate

```python
{code}
```

Evaluate this code against all 6 criteria. Be strict and thorough."""

    print("\n=== Titans Architecture Grading (Claude Sonnet 4.5) ===")
    print("Sending code to evaluator...")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": eval_prompt
            }]
        )

        evaluation = response.content[0].text
        print("\n--- Evaluation ---")
        print(evaluation)

        # Parse the result
        eval_lower = evaluation.lower()

        # Count passes
        pass_count = eval_lower.count(": pass")

        # Check for final result
        if "result: pass" in eval_lower and pass_count >= 6:
            print(f"\nPASS: {pass_count}/6 criteria passed")
            return True
        else:
            print(f"\nFAIL: Only {pass_count}/6 criteria passed (need 6/6)")
            return False

    except Exception as e:
        print(f"FAIL: Error during evaluation: {e}")
        return False
