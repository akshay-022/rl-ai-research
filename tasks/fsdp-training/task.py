"""
FSDP Training Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer using LLM-as-a-judge
"""

import ast
import re
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
2. **Wrap the model with FSDP**: Use FullyShardedDataParallel with a custom wrapping policy
3. **Use DistributedSampler**: Ensure each GPU gets different data
4. **Handle checkpointing correctly**: Only save on rank 0, use proper FSDP state dict methods
5. **Clean up**: Destroy the process group at the end

## Requirements

- Use `torch.distributed.fsdp.FullyShardedDataParallel`
- **IMPORTANT**: Use `lambda_auto_wrap_policy` with a custom lambda function that wraps modules based on:
  - Module type: wrap all `TransformerBlock` layers
  - Parameter count: wrap any module with more than 1,000,000 parameters
  - Do NOT use `transformer_auto_wrap_policy` - implement the logic yourself with `lambda_auto_wrap_policy`
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


def _strip_markdown_wrapper(code: str) -> str:
    """Remove markdown code blocks if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _check_code_executes(code: str) -> tuple[bool, str]:
    """
    Check if the code can be parsed.
    We do a syntax check only - import validation is too environment-dependent.
    The LLM-as-a-judge will evaluate if the imports are correct.
    """
    # Syntax check only
    try:
        ast.parse(code)
        return True, "Code syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


# Grading rubric for LLM-as-a-judge
GRADING_RUBRIC = """You are evaluating an FSDP (Fully Sharded Data Parallel) training script conversion.

The task was to convert a single-GPU PyTorch training script to use FSDP for distributed training.

## REQUIRED COMPONENTS - Evaluate each carefully

### 1. Distributed Initialization (1 point)
- Must call `dist.init_process_group()` with appropriate backend (typically "nccl")
- Should handle LOCAL_RANK environment variable to set the correct device

### 2. FSDP Model Wrapping (2 points)
- Must import and use `FullyShardedDataParallel` (FSDP) from `torch.distributed.fsdp`
- Model must be wrapped with FSDP (1 point)
- Must use `lambda_auto_wrap_policy` with a custom lambda function that:
  - Wraps TransformerBlock layers by type, OR
  - Wraps modules with more than 1,000,000 parameters
  - (NOT transformer_auto_wrap_policy - must be custom lambda) (1 point)

### 3. Data Distribution (2 points)
- Must use `DistributedSampler` to ensure each GPU gets different data (1 point)
- Must call `sampler.set_epoch(epoch)` in the training loop for proper shuffling (1 point)

### 4. Checkpoint Handling (2 points)
- Must save checkpoints only on rank 0 (check for `rank == 0` or `get_rank() == 0`) (1 point)
- Must use FSDP state dict handling (`FSDP.state_dict_type` context manager with `StateDictType` and `FullStateDictConfig`) (1 point)

### 5. Cleanup (1 point)
- Must call `dist.destroy_process_group()` at the end

### 6. Optimizer Creation (1 point)
- Optimizer must be created AFTER the model is wrapped with FSDP (this is critical for FSDP)

### 7. Overall Correctness (1 point)
- The code should be logically correct and would work if executed with torchrun
- Training loop structure should be preserved
- No obvious bugs that would cause the training to fail

## SCORING

Total: 10 points

For each component, award points ONLY if the implementation is correct and complete.
Be STRICT - partial implementations should not receive full points.

## Response Format

Evaluate each component and provide:
1. PASS or FAIL for each sub-item
2. Brief explanation of what you found

```
1. Distributed Initialization: PASS/FAIL (0-1 points)
   - init_process_group: [found/missing]
   - LOCAL_RANK handling: [found/missing]
   - Reasoning: [explanation]

2. FSDP Model Wrapping: PASS/FAIL (0-2 points)
   - FSDP import and usage: [found/missing]
   - lambda_auto_wrap_policy with custom lambda: [found/missing]
   - Parameter count or TransformerBlock check: [found/missing]
   - Reasoning: [explanation]

3. Data Distribution: PASS/FAIL (0-2 points)
   - DistributedSampler: [found/missing]
   - set_epoch call: [found/missing]
   - Reasoning: [explanation]

4. Checkpoint Handling: PASS/FAIL (0-2 points)
   - Rank 0 only saving: [found/missing]
   - FSDP state_dict_type context: [found/missing]
   - Reasoning: [explanation]

5. Cleanup: PASS/FAIL (0-1 points)
   - destroy_process_group: [found/missing]
   - Reasoning: [explanation]

6. Optimizer Creation: PASS/FAIL (0-1 points)
   - Created after FSDP wrap: [yes/no]
   - Reasoning: [explanation]

7. Overall Correctness: PASS/FAIL (0-1 points)
   - Would code run correctly: [yes/no]
   - Reasoning: [explanation]

TOTAL: X/10 points
RESULT: PASS (if >= 8/10) / FAIL (if < 8/10)
```

Be thorough but fair. The code doesn't need to be perfect, but must correctly implement FSDP training."""


def grading_func(result: Any) -> bool:
    """
    Validates the FSDP training script using:
    1. Code execution check (syntax + imports)
    2. LLM-as-a-judge for semantic correctness

    Returns:
        True if the script meets requirements, False otherwise
    """
    import anthropic
    import os
    from dotenv import load_dotenv

    # Load .env from root folder
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    if not result or not isinstance(result, str):
        print("FAIL: No code submitted or not a string")
        return False

    code = _strip_markdown_wrapper(result)

    if len(code) < 200:
        print("FAIL: Code too short to be a valid FSDP implementation")
        return False

    # Step 1: Check code can execute (syntax + imports)
    print("\n=== Step 1: Code Execution Check ===")
    exec_ok, exec_msg = _check_code_executes(code)
    print(f"  {exec_msg}")

    if not exec_ok:
        print(f"FAIL: {exec_msg}")
        return False

    print("  ✓ Code syntax and imports validated")

    # Step 2: LLM-as-a-judge evaluation
    print("\n=== Step 2: LLM-as-a-Judge Evaluation (Claude Sonnet) ===")
    print("  Sending code to evaluator...")

    client = anthropic.Anthropic()

    eval_prompt = GRADING_RUBRIC + f"""

## Code to Evaluate

```python
{code}
```

Evaluate this FSDP implementation against all criteria. Be strict but fair."""

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

        # Look for the final RESULT line
        if "result: pass" in eval_lower:
            print("\n✓ PASS: LLM evaluator approved the implementation")
            return True
        elif "result: fail" in eval_lower:
            print("\n✗ FAIL: LLM evaluator found issues with the implementation")
            return False
        else:
            # Fallback: try to parse the score
            score_match = re.search(r"total:\s*(\d+)/10", eval_lower)
            if score_match:
                score = int(score_match.group(1))
                if score >= 8:
                    print(f"\n✓ PASS: Score {score}/10 meets threshold (8/10)")
                    return True
                else:
                    print(f"\n✗ FAIL: Score {score}/10 below threshold (8/10)")
                    return False

            # If we can't parse, assume fail
            print("\n✗ FAIL: Could not parse evaluation result")
            return False

    except Exception as e:
        print(f"FAIL: Error during LLM evaluation: {e}")
        return False
