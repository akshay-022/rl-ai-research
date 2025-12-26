"""
Memory Optimization Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
"""

import os
import re
import ast
from collections.abc import Callable
from typing import Any, TypedDict
from pathlib import Path


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Tool for submitting the final optimized training script."""
    return {"answer": answer, "submitted": True}


# Get the starter code to include in the prompt
TASK_DIR = Path(__file__).parent
STARTER_CODE_PATH = TASK_DIR / "train_baseline.py"
MODEL_CODE_PATH = TASK_DIR / "model.py"

with open(STARTER_CODE_PATH) as f:
    STARTER_CODE = f.read()

with open(MODEL_CODE_PATH) as f:
    MODEL_CODE = f.read()


# The challenge prompt
PROMPT = f"""You are a machine learning engineer. Your task is to optimize the memory usage of a PyTorch training script.

## Baseline Training Script (train_baseline.py)

```python
{STARTER_CODE}
```

## Model Definition (model.py)

```python
{MODEL_CODE}
```

## Your Task

Optimize the training script to reduce GPU memory usage. Apply memory optimization techniques you know.

Some common techniques include: mixed precision training, gradient checkpointing, efficient attention, gradient accumulation, optimizer optimizations, memory management, etc.

## Important Notes

- For mixed precision: `from torch.cuda.amp import autocast, GradScaler` then `with autocast():` - do NOT pass device_type to autocast
- Do NOT modify the model architecture in model.py - only optimize the training script
- Keep all the original command line arguments working

## Requirements

- Your code must be valid, runnable Python
- Apply at least 5 different memory optimization techniques
- The script must still train the model correctly

Submit your complete optimized training script using submit_answer.
"""


# Tool definitions
SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your complete optimized training script as a string",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete optimized training script code"
            }
        },
        "required": ["answer"],
    },
}

TOOLS = [SUBMIT_TOOL]

TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "submit_answer": submit_answer_tool,
}


def check_code_has_pattern(code: str, pattern: str, description: str) -> tuple[bool, str]:
    """Check if code contains a regex pattern."""
    if re.search(pattern, code, re.MULTILINE | re.DOTALL):
        return True, f"✓ {description}"
    return False, f"✗ Missing: {description}"


def test_code_runs(code: str) -> tuple[bool, str]:
    """
    Test if the generated code actually runs without errors.

    Writes the code to a temp file in the task directory (so imports work),
    then runs it with minimal settings to verify it compiles and executes.
    """
    import subprocess
    import sys

    # Create temp file in task directory so 'from model import ...' works
    temp_path = TASK_DIR / "temp_optimized_train.py"

    try:
        # Write the code to temp file
        with open(temp_path, "w") as f:
            f.write(code)

        # Run with minimal settings: 1 epoch, tiny dataset, small model
        # Use a timeout to prevent hanging
        # Use sys.executable to get the current Python interpreter
        result = subprocess.run(
            [
                sys.executable, str(temp_path),
                "--epochs", "1",
                "--batch-size", "2",
                "--num-samples", "10",
                "--model-size", "small",
            ],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(TASK_DIR),
        )

        if result.returncode == 0:
            return True, "Code runs successfully"
        else:
            # Extract the key error message
            stderr = result.stderr
            # Find the actual error line
            error_lines = [l for l in stderr.split('\n') if l.strip()]
            if error_lines:
                # Get last few lines which usually contain the actual error
                error_msg = '\n'.join(error_lines[-5:])
            else:
                error_msg = stderr[:500] if stderr else "Unknown error"
            return False, f"Runtime error:\n{error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Code timed out (>120s)"
    except Exception as e:
        return False, f"Failed to run code: {str(e)}"
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def grading_func(result: Any) -> bool:
    """
    Validates the memory-optimized training script.

    Checks:
    1. Code runs without errors
    2. At least 5 memory optimization techniques are applied

    Returns:
        True if the script meets all requirements, False otherwise
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

    # 1. Check valid Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"FAIL: Syntax error in submitted code: {e}")
        return False

    # 2. Test that code actually runs
    print("\n=== Runtime Test ===")
    runs_ok, run_msg = test_code_runs(code)
    if runs_ok:
        print(f"✓ {run_msg}")
    else:
        print(f"✗ {run_msg}")
        return False

    # 3. Count optimization techniques
    print("\n=== Optimization Detection ===\n")

    optimizations = [
        (r"(torch\.cuda\.amp|autocast|GradScaler|torch\.amp)", "Mixed Precision (AMP)"),
        (r"(checkpoint|torch\.utils\.checkpoint|checkpoint_sequential|gradient_checkpointing)", "Gradient Checkpointing"),
        (r"(scaled_dot_product_attention|F\.scaled_dot_product_attention|sdpa|flash_attention|FlashAttention)", "SDPA/FlashAttention"),
        (r"(accumulation_steps\s*[=:]|gradient_accumulation|%\s*accumulation|accum_steps\s*=)", "Gradient Accumulation"),
        (r"(bitsandbytes|bnb\.optim|8bit|Adafactor|fused\s*=\s*True|foreach\s*=\s*True|set_to_none\s*=\s*True)", "Memory-efficient Optimizer"),
        (r"(torch\.cuda\.empty_cache\(\)|del\s+(outputs|loss|logits|activations)|\.detach\(\)\.cpu\(\))", "Explicit Memory Management"),
        (r"(offload.*cpu|pin_memory\s*=\s*True|\.cpu\(\).*\.to\(|cpu_offload)", "CPU Offloading / Pin Memory"),
        (r"(micro_batch_size\s*=|effective_batch_size|batch_size\s*//\s*accum|batch_size\s*/\s*accumulation)", "Micro-batching"),
        (r"(inplace\s*=\s*True|\.add_\(|\.mul_\(|\.zero_\(|grad\.add_)", "In-place Operations"),
        (r"(torch\.compile\s*\(|model\s*=.*compile|compiled_model)", "torch.compile"),
        (r"(\.to\(memory_format\s*=|channels_last|contiguous_format\s*=)", "Memory Format Optimization"),
    ]

    count = 0
    for pattern, name in optimizations:
        found = bool(re.search(pattern, code, re.MULTILINE | re.DOTALL))
        status = "✓" if found else "✗"
        print(f"{status} {name}")
        if found:
            count += 1

    print(f"\nTotal optimizations: {count}/11")

    # 4. Check if enough optimizations
    print("\n=== Final Evaluation ===")

    if count >= 5:
        print(f"PASS: Applied {count} memory optimizations (≥5 required)")
        return True
    else:
        print(f"FAIL: Only {count} optimizations detected (need at least 5)")
        return False
