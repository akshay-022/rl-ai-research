"""
Neural Long-Term Memory Implementation Task

This is a single-shot implementation task that:
- Asks the agent to add a neural memory module to a Transformer
- Provides minimal guidance (just requirements, no implementation hints)
- Grades strictly on 6 criteria that match the Titans architecture

Target pass rate: 10-40%
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
PROMPT = f"""You are a Senior AI Architect. Your task is to add a **Neural Long-Term Memory** module to the Transformer below.

## Starter Code
```python
{STARTER_CODE}
```

## Task

Standard attention has quadratic cost and a fixed context window. Your goal is to add an auxiliary memory system that can store and retrieve information beyond the attention window.

Think about how biological memory works:
- The brain doesn't remember everything equally - it prioritizes surprising or novel information
- Old memories fade over time unless reinforced
- Memory involves associating concepts together (binding keys to values)
- We blend immediate context with long-term knowledge

When does a human update their "brain weights"? Your memory system should capture this principle.

Requirements:
1. Create a `NeuralMemory` module that maintains a memory matrix M (shape: d_mem x d_mem)
2. Process tokens recurrently (one at a time), updating M at each step
3. Use outer products to store associations: M += V ⊗ K^T binds values to keys
4. Preserve the input/output signature (`input_ids` -> `logits`)

Important: Make parameters learnable (nn.Parameter) rather than hardcoding values. Use sigmoid activations for gating. The model should be able to learn the right behavior through training.

Submit your complete, runnable Python code.
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


def test_code_execution(code: str) -> tuple[bool, str]:
    """
    Execute the submitted code and verify it runs correctly.

    Returns:
        tuple of (success: bool, message: str)
    """
    import subprocess
    import tempfile

    # Create a test script that imports and runs the model
    test_script = f'''
{code}

# Test the model
import torch

try:
    # Instantiate with smaller params for quick test
    model = SimpleTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=256)
    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 32))  # batch=2, seq_len=32

    # Forward pass
    with torch.no_grad():
        output = model(input_ids)

    # Verify output format
    assert "logits" in output, "Output must contain 'logits' key"
    assert output["logits"].shape == (2, 32, 1000), f"Expected shape (2, 32, 1000), got {{output['logits'].shape}}"

    print("EXECUTION_SUCCESS")
except Exception as e:
    print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}")
'''

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_path = f.name

    try:
        import sys
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout + result.stderr

        if "EXECUTION_SUCCESS" in output:
            return True, "Code executes successfully and produces correct output shape"
        elif "EXECUTION_ERROR" in output:
            error_line = [l for l in output.split('\n') if 'EXECUTION_ERROR' in l]
            return False, error_line[0] if error_line else "Unknown execution error"
        else:
            return False, f"Unexpected output: {output[:500]}"

    except subprocess.TimeoutExpired:
        return False, "Code execution timed out (>30s)"
    except Exception as e:
        return False, f"Failed to run code: {e}"
    finally:
        import os
        os.unlink(temp_path)


def grading_func(result: Any) -> bool:
    """
    Validates the Neural Memory implementation:
    1. First checks if the code executes correctly
    2. Then uses Claude Sonnet 4.5 to evaluate architecture criteria

    Requires both execution success AND 6/6 criteria to pass.
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

    print("\n=== Neural Memory Task Grading ===")

    # Step 1: Test code execution
    print("\n--- Step 1: Code Execution Test ---")
    exec_success, exec_msg = test_code_execution(code)
    print(f"Execution: {'PASS' if exec_success else 'FAIL'} - {exec_msg}")

    if not exec_success:
        print(f"\nFAIL: Code does not execute correctly")
        return False

    # Step 2: Use Claude Sonnet 4.5 to evaluate architecture
    print("\n--- Step 2: Architecture Evaluation (Claude Sonnet 4.5) ---")

    client = anthropic.Anthropic()

    eval_prompt = f"""{GRADING_RUBRIC}

## Code to Evaluate

```python
{code}
```

Evaluate this code against all 6 criteria. Be strict and thorough."""

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
        print(evaluation)

        # Parse the result
        eval_lower = evaluation.lower()

        # Count passes
        pass_count = eval_lower.count(": pass")

        # Check for final result
        if "result: pass" in eval_lower and pass_count >= 6:
            print(f"\nPASS: Execution OK + {pass_count}/6 criteria passed")
            return True
        else:
            print(f"\nFAIL: Only {pass_count}/6 criteria passed (need 6/6)")
            return False

    except Exception as e:
        print(f"FAIL: Error during evaluation: {e}")
        return False


# --- MULTI-RUN EVALUATION SUPPORT ---

def run_single_evaluation(model: str = "claude-haiku-4-5", verbose: bool = True) -> dict:
    """
    Run a single evaluation with a multi-turn agent that can think before submitting.

    The agent can take multiple turns to reason about the architecture,
    but only the final submitted code is evaluated.

    Args:
        model: The model to test
        verbose: Whether to print detailed output

    Returns:
        dict with evaluation results
    """
    client = anthropic.Anthropic()

    if verbose:
        print(f"\n{'='*60}")
        print(f"NEURAL MEMORY TASK EVALUATION")
        print(f"Model: {model}")
        print(f"{'='*60}")

    messages = [{
        "role": "user",
        "content": PROMPT
    }]

    submitted_code = None
    max_turns = 5  # Allow up to 5 turns of thinking

    for turn in range(max_turns):
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")

        response = client.messages.create(
            model=model,
            max_tokens=16384,
            tools=TOOLS,
            messages=messages
        )

        # Check if model submitted code
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                submitted_code = block.input.get("answer")
                if verbose:
                    print("Agent submitted code.")
                break

        if submitted_code:
            break

        # If no submission yet, check for text reasoning and continue
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        if verbose and text_content:
            # Print abbreviated thinking
            preview = text_content[:500] + "..." if len(text_content) > 500 else text_content
            print(f"Agent thinking: {preview}")

        # If model stopped without tool use, prompt it to submit
        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Please submit your complete implementation using the submit_answer tool."
            })
        else:
            # Model used a tool but not submit_answer - shouldn't happen with our single tool
            break

    if not submitted_code:
        if verbose:
            print("FAIL: No code submission after max turns")
        return {
            "model": model,
            "passed": False,
            "reason": "No code submission",
            "code": None
        }

    # Grade the submission
    passed = grading_func(submitted_code)

    return {
        "model": model,
        "passed": passed,
        "code": submitted_code[:500] + "..." if len(submitted_code) > 500 else submitted_code
    }


def run_multiple_evaluations(model: str = "claude-haiku-4-5", num_runs: int = 5) -> dict:
    """
    Run multiple evaluations and compute pass rate.

    Args:
        model: The model to test
        num_runs: Number of evaluation runs

    Returns:
        dict with aggregated statistics
    """
    import concurrent.futures

    print(f"\n{'='*60}")
    print(f"TITANS COMBINED TASK - {num_runs} RUNS")
    print(f"Model: {model}")
    print(f"{'='*60}")

    def run_quiet(run_id: int) -> dict:
        result = run_single_evaluation(model, verbose=False)
        result["run_id"] = run_id
        return result

    # Run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_runs, 5)) as executor:
        futures = [executor.submit(run_quiet, i) for i in range(num_runs)]
        results = [f.result() for f in futures]

    # Aggregate
    pass_count = sum(1 for r in results if r["passed"])
    pass_rate = pass_count / num_runs

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"Run {r['run_id']+1}: {status}")

    print(f"\nPass Rate: {pass_count}/{num_runs} ({pass_rate*100:.0f}%)")

    # Check if in target range
    if 0.10 <= pass_rate <= 0.40:
        print("✓ In target range (10-40%)")
    elif pass_rate < 0.10:
        print("⚠ Below target range - task may be too hard")
    else:
        print("⚠ Above target range - task may be too easy")

    return {
        "model": model,
        "num_runs": num_runs,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "results": results
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        model = sys.argv[3] if len(sys.argv) > 3 else "claude-haiku-4-5"
        run_multiple_evaluations(model, num_runs)
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"
        run_single_evaluation(model)
