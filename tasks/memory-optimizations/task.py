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
PROMPT = f"""You are an expert machine learning engineer specializing in GPU memory optimization. Your goal is to create the most memory-efficient training script possible.

Your task is to optimize the memory usage of a PyTorch training script for a deep transformer model.

## Baseline Training Script (train_baseline.py)

```python
{STARTER_CODE}
```

## Model Definition (model.py)

```python
{MODEL_CODE}
```

## Your Task

Optimize the training script to MAXIMIZE memory savings. Apply as many memory optimization techniques as possible. The more optimizations you apply, the better your score.

## Required Optimizations (you MUST include ALL of these)

1. **Mixed Precision Training (AMP)**: Use torch.cuda.amp.autocast and GradScaler
2. **Gradient Checkpointing**: Use torch.utils.checkpoint to trade compute for memory
3. **Memory-Efficient Attention**: Use F.scaled_dot_product_attention (SDPA/FlashAttention)

## Additional Optimizations (apply AS MANY as possible)

You MUST apply at least 2 of these additional techniques. The more you apply, the better:

4. **Gradient Accumulation**: Use accumulation_steps to simulate larger batches with smaller micro-batches
5. **Memory-efficient Optimizer**: Use set_to_none=True in zero_grad(), or fused=True/foreach=True in optimizer
6. **Explicit Memory Management**: Use torch.cuda.empty_cache() and del intermediate tensors (del outputs, del loss)
7. **CPU Offloading / Pin Memory**: Use pin_memory=True in DataLoader
8. **Micro-batching**: Define micro_batch_size or effective_batch_size variables
9. **In-place Operations**: Use inplace=True for activations where possible
10. **torch.compile**: Use torch.compile() for kernel fusion and optimization
11. **Memory Format**: Use channels_last memory format if applicable

## Requirements

Your optimized script must:
1. Include ALL 3 required optimizations (AMP, Gradient Checkpointing, SDPA)
2. Include AT LEAST 2 additional optimizations from the list above
3. Be valid, runnable Python code
4. Include brief comments explaining each optimization

## IMPORTANT

Do NOT stop at just the minimum. Apply every optimization technique you can. A script with 5+ total optimizations is good. A script with 7+ optimizations is excellent.

## Submission

Submit your complete optimized training script as a string using submit_answer.
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
    import tempfile

    # Create temp file in task directory so 'from model import ...' works
    temp_path = TASK_DIR / "temp_optimized_train.py"

    try:
        # Write the code to temp file
        with open(temp_path, "w") as f:
            f.write(code)

        # Run with minimal settings: 1 epoch, tiny dataset, small model
        # Use a timeout to prevent hanging
        result = subprocess.run(
            [
                "python", str(temp_path),
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

    Checks for various memory optimization techniques.
    The solution must beat a baseline that already has:
    - Mixed Precision (AMP)
    - Gradient Checkpointing
    - SDPA (Scaled Dot Product Attention)

    To pass, the solution needs these PLUS additional optimizations.

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

    # === BASELINE OPTIMIZATIONS (Required - these are what we compare against) ===
    baseline_checks = []

    # Mixed Precision (AMP)
    baseline_checks.append(check_code_has_pattern(
        code,
        r"(torch\.cuda\.amp|autocast|GradScaler|torch\.amp)",
        "Mixed Precision Training (AMP)"
    ))

    # Gradient Checkpointing
    baseline_checks.append(check_code_has_pattern(
        code,
        r"(checkpoint|torch\.utils\.checkpoint|checkpoint_sequential|gradient_checkpointing)",
        "Gradient Checkpointing"
    ))

    # SDPA / FlashAttention
    baseline_checks.append(check_code_has_pattern(
        code,
        r"(scaled_dot_product_attention|F\.scaled_dot_product_attention|sdpa|flash_attention|FlashAttention)",
        "Scaled Dot Product Attention (SDPA/FlashAttention)"
    ))

    # === ADDITIONAL OPTIMIZATIONS (Need at least 2 to beat baseline) ===
    additional_checks = []

    # Gradient Accumulation (must be used in code, not just mentioned)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(accumulation_steps\s*[=:]|gradient_accumulation|%\s*accumulation|accum_steps\s*=)",
        "Gradient Accumulation"
    ))

    # Memory-efficient optimizer states (8-bit Adam, Adafactor, etc.)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(bitsandbytes|bnb\.optim|8bit|Adafactor|fused\s*=\s*True|foreach\s*=\s*True|set_to_none\s*=\s*True)",
        "Memory-efficient Optimizer"
    ))

    # Activation memory optimization (del intermediate, empty_cache, etc.)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(torch\.cuda\.empty_cache\(\)|del\s+(outputs|loss|logits|activations)|\.detach\(\)\.cpu\(\))",
        "Explicit Memory Management"
    ))

    # CPU offloading / pin memory (actual usage, not just method definition)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(offload.*cpu|pin_memory\s*=\s*True|\.cpu\(\).*\.to\(|cpu_offload)",
        "CPU Offloading / Pin Memory"
    ))

    # Smaller batch with accumulation (actual micro-batch variable usage)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(micro_batch_size\s*=|effective_batch_size|batch_size\s*//\s*accum|batch_size\s*/\s*accumulation)",
        "Micro-batching Strategy"
    ))

    # In-place operations (actual usage in optimizer or tensor ops)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(inplace\s*=\s*True|\.add_\(|\.mul_\(|\.zero_\(|grad\.add_)",
        "In-place Operations"
    ))

    # Model compilation / torch.compile (actual usage)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(torch\.compile\s*\(|model\s*=.*compile|compiled_model)",
        "torch.compile Optimization"
    ))

    # Channels last memory format (actual usage)
    additional_checks.append(check_code_has_pattern(
        code,
        r"(\.to\(memory_format\s*=|channels_last|contiguous_format\s*=)",
        "Memory Format Optimization"
    ))

    # Print results
    print("\n=== Memory Optimization Task Grading ===\n")

    print("--- Baseline Optimizations (Required) ---")
    baseline_passed = 0
    for success, msg in baseline_checks:
        print(msg)
        if success:
            baseline_passed += 1

    print(f"\nBaseline Score: {baseline_passed}/{len(baseline_checks)}")

    print("\n--- Additional Optimizations (Need 2+ to beat baseline) ---")
    additional_passed = 0
    for success, msg in additional_checks:
        print(msg)
        if success:
            additional_passed += 1

    print(f"\nAdditional Score: {additional_passed}/{len(additional_checks)}")

    # === GRADING LOGIC ===
    # Must have ALL 3 baseline optimizations
    # Must have at least 2 additional optimizations to "beat" the baseline

    all_baseline = baseline_passed == len(baseline_checks)
    enough_additional = additional_passed >= 2

    print("\n=== Final Evaluation ===")

    if not all_baseline:
        print(f"FAIL: Missing baseline optimizations ({baseline_passed}/{len(baseline_checks)})")
        print("       Your solution must include: AMP, Gradient Checkpointing, and SDPA")
        return False

    if not enough_additional:
        print(f"FAIL: Not enough additional optimizations ({additional_passed}/2 required)")
        print("       You matched the baseline but didn't beat it!")
        print("       Add more optimizations like: gradient accumulation, 8-bit optimizers,")
        print("       explicit memory management, CPU offloading, torch.compile, etc.")
        return False

    total_optimizations = baseline_passed + additional_passed
    print(f"PASS: Applied {total_optimizations} memory optimizations!")
    print(f"      Baseline: {baseline_passed}/3 ✓")
    print(f"      Additional: {additional_passed}+ ✓ (beats baseline!)")
    return True
