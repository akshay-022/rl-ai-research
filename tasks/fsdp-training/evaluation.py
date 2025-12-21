"""
FSDP Training Task Evaluation Runner

Run this file to test the FSDP training task with an agent.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from task import PROMPT, TOOLS, TOOL_HANDLERS, grading_func

MAX_TOKENS = 8000  # FSDP code can be lengthy


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        if response.stop_reason == "max_tokens":
            print(f"Warning: Model reached max_tokens limit ({MAX_TOKENS})")

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    # Truncate long text output
                    text = content.text
                    if len(text) > 500:
                        text = text[:500] + "..."
                    print(f"Assistant: {text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                        if verbose:
                            # Show first 500 chars of submitted code
                            code_preview = submitted_answer[:500] if len(submitted_answer) > 500 else submitted_answer
                            print(f"Submitted code preview:\n{code_preview}...")
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )
                        if verbose and result.get("error"):
                            print(f"  Error: {result['error']}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    })

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\n{'='*60}")
                    print("SUBMITTED FSDP TRAINING SCRIPT")
                    print("="*60)
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    """Run a single test iteration."""
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=PROMPT,
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        max_steps=10,
        verbose=verbose,
    )

    success = grading_func(result)

    display_result = "Code submitted" if result else "No code submitted"

    if success:
        print(f"✓ Run {run_id}: SUCCESS")
    else:
        print(f"✗ Run {run_id}: FAILURE - {display_result}")

    return run_id, success, result


async def main(num_runs: int = 1, verbose: bool = True):
    """Run evaluation tests."""
    print(f"Running {num_runs} test iteration(s)...")
    print("=" * 60)

    results = []
    for i in range(num_runs):
        result = await run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            verbose=verbose,
        )
        results.append(result)

    successes = sum(success for _, success, _ in results)
    pass_rate = (successes / num_runs) * 100

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FSDP training task evaluation")
    parser.add_argument("-n", "--num-runs", type=int, default=1, help="Number of test runs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    asyncio.run(main(num_runs=args.num_runs, verbose=not args.quiet))
