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


def _count_optimizations_with_llm(code: str) -> int:
    """
    Use an LLM to count the number of distinct memory optimization techniques in the code.
    Returns an integer count.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    try:
        import anthropic
        client = anthropic.Anthropic()

        prompt = """Analyze the following PyTorch training script and count how many DISTINCT memory optimization techniques are used.

Common memory optimization techniques include (but are not limited to):
1. Mixed Precision Training (AMP) - using torch.cuda.amp, autocast, GradScaler
2. Gradient Checkpointing - using torch.utils.checkpoint
3. Flash Attention / SDPA - using scaled_dot_product_attention
4. Gradient Accumulation - accumulating gradients over multiple steps
5. Memory-efficient Optimizers - using fused=True, foreach=True, set_to_none=True, 8-bit optimizers
6. Explicit Memory Management - torch.cuda.empty_cache(), del statements, .detach().cpu()
7. CPU Offloading / Pin Memory - offloading to CPU, pin_memory=True
8. Micro-batching - splitting batches into smaller micro-batches
9. In-place Operations - using inplace=True, .add_(), .mul_(), .zero_()
10. torch.compile - using torch.compile() for optimization
11. Memory Format Optimization - using channels_last format

Count each distinct technique only ONCE, even if it appears multiple times in the code.
Only count techniques that are actually implemented and used, not just mentioned in comments.

YOUR RESPONSE MUST BE A SINGLE INTEGER AND NOTHING ELSE.
Do not include any explanation, just output the number.

CODE:
```python
{code}
```"""

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": prompt.format(code=code)
            }]
        )

        result = response.content[0].text.strip()
        count = int(result)
        return count

    except ValueError as e:
        print(f"  LLM returned non-integer: {result}")
        return 0
    except Exception as e:
        print(f"  LLM optimization count error: {e}")
        return 0


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
    1. Code is valid Python syntax
    2. Code runs without errors
    3. At least 6 memory optimization techniques are applied

    Returns:
        True if the submission passes all checks, False otherwise
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

    # 3. Count optimization techniques using LLM
    print("\n=== Optimization Detection ===\n")

    count = _count_optimizations_with_llm(code)
    print(f"Total optimizations detected: {count}")

    # 4. Check if enough optimizations
    print("\n=== Final Evaluation ===")

    if count >= 5:
        print(f"PASS: Applied {count} memory optimizations (≥5 required)")
        return True
    else:
        print(f"FAIL: Only {count} optimizations detected (need at least 5)")
        return False
