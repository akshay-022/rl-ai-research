"""
FSDP Training Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
"""

import os
import re
import ast
import tempfile
import subprocess
from collections.abc import Callable
from typing import Any, TypedDict
from pathlib import Path


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Tool for submitting the final FSDP training script."""
    return {"answer": answer, "submitted": True}


# Get the starter code to include in the prompt
TASK_DIR = Path(__file__).parent
STARTER_CODE_PATH = TASK_DIR / "train_single_gpu.py"
MODEL_CODE_PATH = TASK_DIR / "model.py"

with open(STARTER_CODE_PATH) as f:
    STARTER_CODE = f.read()

with open(MODEL_CODE_PATH) as f:
    MODEL_CODE = f.read()


# The challenge prompt
PROMPT = f"""You are a machine learning engineer. Your task is to convert a single-GPU PyTorch training script to use Fully Sharded Data Parallel (FSDP) for distributed training.

## Starter Code (train_single_gpu.py)

```python
{STARTER_CODE}
```

## Model Definition (model.py)

```python
{MODEL_CODE}
```

## Your Task

Convert the training script to use PyTorch FSDP. Your solution must:

1. **Initialize distributed training**: Set up the process group with NCCL backend
2. **Wrap the model with FSDP**: Use FullyShardedDataParallel with proper wrapping policy
3. **Use DistributedSampler**: Ensure each GPU gets different data
4. **Handle checkpointing correctly**: Only save on rank 0, use proper FSDP state dict methods
5. **Clean up**: Destroy the process group at the end

## Requirements

- Use `torch.distributed.fsdp.FullyShardedDataParallel`
- Use `transformer_auto_wrap_policy` to wrap TransformerBlock layers
- Create the optimizer AFTER wrapping with FSDP
- Call `sampler.set_epoch(epoch)` in the training loop
- Only print/log on rank 0 to avoid duplicate output
- Use `FSDP.state_dict_type` context manager for saving checkpoints

## Submission

Submit your complete modified training script as a string using submit_answer.
The script should be runnable with: `torchrun --nproc_per_node=2 train_fsdp.py`
"""


# Tool definitions
SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your complete FSDP training script as a string",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete FSDP training script code"
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


def grading_func(result: Any) -> bool:
    """
    Validates the FSDP training script.

    Checks:
    1. Syntax is valid Python
    2. Has distributed initialization
    3. Uses FSDP wrapper
    4. Uses DistributedSampler
    5. Has proper checkpoint saving
    6. Creates optimizer after FSDP wrap
    7. Calls set_epoch on sampler
    8. Cleans up process group

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
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    # 1. Check valid Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"FAIL: Syntax error in submitted code: {e}")
        return False

    checks = []

    # 2. Distributed initialization
    checks.append(check_code_has_pattern(
        code,
        r"dist\.init_process_group|init_process_group\s*\(",
        "dist.init_process_group() call"
    ))

    # 3. FSDP import and usage
    checks.append(check_code_has_pattern(
        code,
        r"from\s+torch\.distributed\.fsdp\s+import.*FullyShardedDataParallel|FSDP\s*\(",
        "FSDP import/usage"
    ))

    checks.append(check_code_has_pattern(
        code,
        r"FSDP\s*\(|FullyShardedDataParallel\s*\(",
        "Model wrapped with FSDP"
    ))

    # 4. DistributedSampler
    checks.append(check_code_has_pattern(
        code,
        r"DistributedSampler\s*\(",
        "DistributedSampler usage"
    ))

    # 5. Wrapping policy for TransformerBlock
    checks.append(check_code_has_pattern(
        code,
        r"transformer_auto_wrap_policy|auto_wrap_policy.*TransformerBlock|TransformerBlock.*auto_wrap",
        "Auto wrap policy for TransformerBlock"
    ))

    # 6. set_epoch call
    checks.append(check_code_has_pattern(
        code,
        r"sampler\.set_epoch\s*\(|\.set_epoch\s*\(\s*epoch",
        "sampler.set_epoch() call"
    ))

    # 7. Rank 0 checkpoint saving
    checks.append(check_code_has_pattern(
        code,
        r"(get_rank\s*\(\s*\)\s*==\s*0|rank\s*==\s*0).*save|if.*rank.*0.*:.*\n.*save",
        "Checkpoint saving on rank 0 only"
    ))

    # 8. FSDP state dict handling
    checks.append(check_code_has_pattern(
        code,
        r"state_dict_type|StateDictType|FullStateDictConfig",
        "FSDP state dict handling"
    ))

    # 9. Process group cleanup
    checks.append(check_code_has_pattern(
        code,
        r"destroy_process_group\s*\(",
        "destroy_process_group() call"
    ))

    # 10. LOCAL_RANK handling
    checks.append(check_code_has_pattern(
        code,
        r"LOCAL_RANK|local_rank",
        "LOCAL_RANK environment variable handling"
    ))

    # Print results
    passed = 0
    total = len(checks)

    print("\n=== FSDP Task Grading ===")
    for success, msg in checks:
        print(msg)
        if success:
            passed += 1

    print(f"\nScore: {passed}/{total}")

    # Require at least 8/10 checks to pass
    # This allows some flexibility while ensuring core concepts are present
    threshold = 8
    if passed >= threshold:
        print(f"PASS: {passed}/{total} checks passed (threshold: {threshold})")
        return True
    else:
        print(f"FAIL: Only {passed}/{total} checks passed (need {threshold})")
        return False
