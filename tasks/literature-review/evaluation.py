"""
Literature Review Evaluation Runner

Run this file to test the literature review agent with Haiku.
Usage:
    python evaluation.py           # Single run with verbose output
    python evaluation.py -n 5      # 5 runs
    python evaluation.py -n 5 -q   # 5 runs, quiet mode
"""

import asyncio
import json
import os
import sys
import textwrap
from collections.abc import Callable
from datetime import datetime
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from dotenv import load_dotenv

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import tools directly (avoid relative import issues)
from tools import (
    WEB_SEARCH_TOOL,
    GET_PAPER_INTRO_TOOL,
    GET_PAPER_METHOD_TOOL,
    GET_PAPER_RESULTS_TOOL,
    web_search,
    get_paper_introduction,
    get_paper_methodology,
    get_paper_results,
)

# ============================================================
# TASK DEFINITION (inline to avoid import issues)
# ============================================================

TOPIC = "giving LLMs long term memory"

PROMPT = f"""You are a research assistant conducting a literature review on: "{TOPIC}"

You have 4 research tools available:
1. web_search - Search for papers by topic
2. get_paper_introduction - Extract the INTRODUCTION section (literature context, related work)
3. get_paper_methodology - Extract the METHODS section (how the approach works)
4. get_paper_results - Extract the RESULTS section (quantitative findings, benchmarks)

Your workflow:
1. Use web_search to find relevant papers for each major approach
2. For important papers, extract the relevant sections:
   - get_paper_introduction for literature context and related work
   - get_paper_results for quantitative findings and performance numbers
   - get_paper_methodology if you need to understand how something works
3. Synthesize everything into a structured literature review

Your final review MUST cover ALL of these major approaches:
- Extended context windows (long-context models like Gemini 1M, efficient attention like Longformer, RMT)
- Retrieval-Augmented Generation (RAG, RETRO, external knowledge stores)
- Summarization and context compression (recursive summarization, context distillation)
- Specialized memory architectures (MemGPT, LongMem, Generative Agents, memory modules)
- Parametric memory (continual learning, model editing like ROME/MEMIT, catastrophic forgetting)

For EACH approach, include:
- How it works (brief explanation)
- Key papers and their contributions
- Reported results/performance improvements (MUST include specific quantitative results)
- Limitations or trade-offs

Use submit_answer to submit your complete literature review.
Be thorough and cite specific papers with their key findings and quantitative results."""


# Submit tool
def submit_answer_tool(answer: Any) -> dict:
    """Tool for submitting the final literature review."""
    return {"answer": answer, "submitted": True}


SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final literature review",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The complete literature review"}},
        "required": ["answer"],
    },
}

# Combined tools and handlers
TOOLS = [WEB_SEARCH_TOOL, GET_PAPER_INTRO_TOOL, GET_PAPER_METHOD_TOOL, GET_PAPER_RESULTS_TOOL, SUBMIT_TOOL]
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "web_search": lambda query, max_results=5: web_search(query, max_results),
    "get_paper_introduction": lambda paper_id: get_paper_introduction(paper_id),
    "get_paper_methodology": lambda paper_id: get_paper_methodology(paper_id),
    "get_paper_results": lambda paper_id: get_paper_results(paper_id),
    "submit_answer": submit_answer_tool,
}


# ============================================================
# GRADING FUNCTION
# ============================================================

def _load_reference_material():
    """Load the reference literature review from the dataset."""
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    memory_path = os.path.join(dataset_dir, 'memory.md')

    if os.path.exists(memory_path):
        with open(memory_path, 'r') as f:
            return f.read()
    return None


REFERENCE_MATERIAL = _load_reference_material()


GRADING_RUBRIC = """You are evaluating a literature review on "giving LLMs long term memory".

## REQUIRED COVERAGE - Check each item carefully

The review MUST cover these 5 major approaches AND mention specific seminal works within each.

### APPROACH 1: Extended Context Windows (2 points total)
Required topics (1 point if ANY mentioned):
- Long-context proprietary models: Claude 100k, GPT-4 128k/32k, Gemini 1.5 (1M tokens)
- Efficient attention: Longformer, BigBird, sparse attention
- Recurrent/segment processing: RMT (Recurrent Memory Transformer)
- Context limitations: "context rot", attention dilution, "lost in the middle"

Required results (1 point if ANY mentioned):
- Gemini 1.5: 99% needle-in-haystack at 1M tokens
- RMT: processed up to 2M tokens
- Any specific benchmark number for long context

### APPROACH 2: Retrieval-Augmented Generation (2 points total)
Required topics (1 point if ANY mentioned):
- RAG architecture: external knowledge stores, vector databases, embedding retrieval
- Key systems: RETRO, RAG (Lewis et al), Atlas

Required results (1 point if ANY mentioned):
- RETRO: 7.5B model outperformed 175B Jurassic-1
- Any quantitative improvement from retrieval augmentation

### APPROACH 3: Summarization & Context Compression (2 points total)
Required topics (1 point if ANY mentioned):
- Recursive/iterative summarization
- Conversation summary memory (e.g., LangChain ConversationSummaryMemory)
- Context distillation, context scaffolding

Required results (1 point if ANY mentioned):
- Improved long-dialogue consistency
- Any specific improvement metric (e.g., +3% BLEU, human preference)

### APPROACH 4: Specialized Memory Architectures (2 points total)
Required topics (1 point if ANY mentioned):
- MemGPT: LLM as OS managing memory, paging context in/out
- LongMem: decoupled memory bank with side network
- Generative Agents: structured memory with reflection
- Key-value memory networks, hierarchical memory, memory graphs

Required results (1 point if ANY mentioned):
- LongMem: 40.5% on ChapterBreak, beat GPT-3 with 313x fewer params
- MemGPT: outperformed vanilla LLMs on long documents
- Generative Agents: believable multi-day agent behavior

### APPROACH 5: Parametric Memory / Weight Updates (2 points total)
Required topics (1 point if ANY mentioned):
- Continual learning, catastrophic forgetting
- Model editing: ROME, MEND, MEMIT
- Fine-tuning for knowledge updates, fast weights

Required results (1 point if ANY mentioned):
- MEMIT: edited 5000+ facts with minimal side effects
- Any quantitative result on knowledge editing or continual learning

## SCORING

Count total points out of 10 (2 per approach: 1 for topics, 1 for results).

For each approach, check:
- Topics covered? (1 point if yes)
- Specific results/numbers mentioned? (1 point if yes)

## Response Format

For each approach, you MUST provide:
1. PASS or FAIL for topics
2. PASS or FAIL for results
3. A detailed explanation of WHY it passed or failed (what was mentioned vs what was missing)

```
APPROACH 1 - Extended Context:
- Topics: PASS/FAIL
  - What was mentioned: [list specific papers/concepts found]
  - What was missing: [list required items not found]
  - Reasoning: [explain why this passes or fails]
- Results: PASS/FAIL
  - Numbers mentioned: [list any quantitative results found]
  - What was missing: [list required numbers not found]
  - Reasoning: [explain why this passes or fails]

APPROACH 2 - RAG/Retrieval:
- Topics: PASS/FAIL
  - What was mentioned: [list specific papers/concepts found]
  - What was missing: [list required items not found]
  - Reasoning: [explain why this passes or fails]
- Results: PASS/FAIL
  - Numbers mentioned: [list any quantitative results found]
  - What was missing: [list required numbers not found]
  - Reasoning: [explain why this passes or fails]

APPROACH 3 - Summarization:
- Topics: PASS/FAIL
  - What was mentioned: [list specific papers/concepts found]
  - What was missing: [list required items not found]
  - Reasoning: [explain why this passes or fails]
- Results: PASS/FAIL
  - Numbers mentioned: [list any quantitative results found]
  - What was missing: [list required numbers not found]
  - Reasoning: [explain why this passes or fails]

APPROACH 4 - Memory Architectures:
- Topics: PASS/FAIL
  - What was mentioned: [list specific papers/concepts found]
  - What was missing: [list required items not found]
  - Reasoning: [explain why this passes or fails]
- Results: PASS/FAIL
  - Numbers mentioned: [list any quantitative results found]
  - What was missing: [list required numbers not found]
  - Reasoning: [explain why this passes or fails]

APPROACH 5 - Parametric Memory:
- Topics: PASS/FAIL
  - What was mentioned: [list specific papers/concepts found]
  - What was missing: [list required items not found]
  - Reasoning: [explain why this passes or fails]
- Results: PASS/FAIL
  - Numbers mentioned: [list any quantitative results found]
  - What was missing: [list required numbers not found]
  - Reasoning: [explain why this passes or fails]

TOTAL: X/10
RESULT: PASS (if >= 7/10) / FAIL (if < 7/10)
```

Be STRICT. Only give points for specific mentions, not vague references."""


def grading_func(result: Any) -> bool:
    """
    Validates the literature review using Claude Sonnet as evaluator.
    """
    import anthropic

    if not result or not isinstance(result, str):
        print("FAIL: No review submitted or not a string")
        return False

    review = result.strip()

    if len(review) < 1000:
        print("FAIL: Review too short (< 1000 characters) for comprehensive coverage")
        return False

    if not REFERENCE_MATERIAL:
        print("WARNING: No reference material found, using basic grading")
        review_lower = review.lower()
        memory_terms = ["memory", "long-term", "retrieval", "context", "storage"]
        method_terms = ["method", "approach", "technique", "architecture", "model"]
        has_memory = any(term in review_lower for term in memory_terms)
        has_methods = any(term in review_lower for term in method_terms)
        return has_memory and has_methods

    client = anthropic.Anthropic()

    ref_material = REFERENCE_MATERIAL
    if len(ref_material) > 20000:
        ref_material = ref_material[:20000] + "\n\n[Reference truncated for brevity...]"

    eval_prompt = GRADING_RUBRIC.format(reference_material=ref_material)
    eval_prompt += f"""

## Literature Review to Evaluate

{review}

Evaluate this review against all 10 criteria. Be STRICT - require specific papers and quantitative results for each approach."""

    print("\n=== Literature Review Grading (Claude Sonnet 4.5) ===")
    print("Sending review to evaluator...")

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

        eval_lower = evaluation.lower()
        pass_count = eval_lower.count(": pass")

        if "result: pass" in eval_lower and pass_count >= 7:
            print(f"\nPASS: {pass_count}/10 criteria passed")
            return True
        else:
            print(f"\nFAIL: Only {pass_count}/10 criteria passed (need 7/10)")
            return False

    except Exception as e:
        print(f"FAIL: Error during evaluation: {e}")
        return False


# ============================================================
# VERBOSE LOGGING HELPERS
# ============================================================

def _print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def _print_subheader(text: str, char: str = "-", width: int = 60):
    """Print a formatted subheader."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def _wrap_text(text: str, width: int = 76, indent: str = "    ") -> str:
    """Wrap text with indentation."""
    if not text:
        return f"{indent}(none)"
    lines = textwrap.wrap(text, width=width - len(indent))
    return "\n".join(f"{indent}{line}" for line in lines)


def _format_paper_result(paper: dict, index: int = None) -> str:
    """Format a paper for display."""
    prefix = f"  [{index}] " if index is not None else "  • "
    title = paper.get("title", "Unknown title")
    year = paper.get("year", "?")
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:2]) + ("..." if len(authors) > 2 else "") if authors else "Unknown authors"
    paper_id = paper.get("paper_id") or paper.get("arxiv_id") or ""

    lines = [
        f"{prefix}{title}",
        f"      Year: {year} | Authors: {author_str}",
    ]
    if paper_id:
        lines.append(f"      ID: {paper_id}")

    abstract = paper.get("abstract")
    if abstract:
        short_abstract = abstract[:200] + "..." if len(abstract) > 200 else abstract
        lines.append(f"      Abstract: {short_abstract}")

    return "\n".join(lines)


def _log_tool_call(tool_name: str, tool_input: dict):
    """Log a tool call with formatted input."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 🔧 TOOL CALL: {tool_name}")
    print(f"{'─' * 50}")

    if tool_name == "web_search":
        print(f"  Query: \"{tool_input.get('query', '')}\"")
        print(f"  Max results: {tool_input.get('max_results', 5)}")
    elif tool_name == "get_paper_introduction":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
        print(f"  (Extracting INTRODUCTION section - literature context)")
    elif tool_name == "get_paper_methodology":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
        print(f"  (Extracting METHODOLOGY section - how approach works)")
    elif tool_name == "get_paper_results":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
        print(f"  (Extracting RESULTS section - quantitative findings)")
    elif tool_name == "submit_answer":
        print(f"  (Submitting final literature review)")
    else:
        print(f"  Input: {json.dumps(tool_input, indent=2)[:500]}")


def _log_tool_result(tool_name: str, result: dict):
    """Log a tool result with formatted output."""
    print(f"\n  📤 RESULT:")

    if result.get("error"):
        print(f"  ❌ Error: {result['error']}")
        return

    if tool_name == "web_search":
        papers = result.get("results", [])
        source = result.get("source", "unknown")
        print(f"  ✓ Found {len(papers)} papers (source: {source})")
        print()
        for i, paper in enumerate(papers, 1):
            print(_format_paper_result(paper, i))
            print()

    elif tool_name == "get_paper_introduction":
        title = result.get("title", "Unknown")
        arxiv_id = result.get("arxiv_id", "")
        introduction = result.get("introduction")

        print(f"  ✓ Paper Introduction Extracted:")
        print(f"      Title: {title}")
        if arxiv_id:
            print(f"      arXiv ID: {arxiv_id}")
        if introduction:
            print(f"\n      📖 INTRODUCTION:")
            print(f"      {'─' * 60}")
            if len(introduction) > 2000:
                print(_wrap_text(introduction[:2000], indent="         "))
                print(f"\n         ... ({len(introduction) - 2000} more chars)")
            else:
                print(_wrap_text(introduction, indent="         "))
        else:
            print(f"\n      ⚠️  Could not extract introduction")

    elif tool_name == "get_paper_methodology":
        title = result.get("title", "Unknown")
        methodology = result.get("methodology")

        print(f"  ✓ Paper Methodology Extracted:")
        print(f"      Title: {title}")
        if methodology:
            print(f"\n      🔧 METHODOLOGY:")
            print(f"      {'─' * 60}")
            if len(methodology) > 2000:
                print(_wrap_text(methodology[:2000], indent="         "))
                print(f"\n         ... ({len(methodology) - 2000} more chars)")
            else:
                print(_wrap_text(methodology, indent="         "))
        else:
            print(f"\n      ⚠️  Could not extract methodology")

    elif tool_name == "get_paper_results":
        title = result.get("title", "Unknown")
        results_text = result.get("results")

        print(f"  ✓ Paper Results Extracted:")
        print(f"      Title: {title}")
        if results_text:
            print(f"\n      📊 RESULTS:")
            print(f"      {'─' * 60}")
            if len(results_text) > 2000:
                print(_wrap_text(results_text[:2000], indent="         "))
                print(f"\n         ... ({len(results_text) - 2000} more chars)")
            else:
                print(_wrap_text(results_text, indent="         "))
        else:
            print(f"\n      ⚠️  Could not extract results")

    else:
        # Generic result display
        print(f"  {json.dumps(result, indent=2)[:800]}")


# ============================================================
# AGENT LOOP
# ============================================================

MAX_TOKENS = 16000  # Larger output for comprehensive literature reviews


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    if verbose:
        _print_header(f"STARTING AGENT LOOP (model: {model})")
        print(f"\nPrompt (first 500 chars):\n{prompt[:500]}...")
        print(f"\nMax steps: {max_steps}")
        print(f"Available tools: {[t['name'] for t in tools]}")

    for step in range(max_steps):
        if verbose:
            _print_subheader(f"STEP {step + 1}/{max_steps}", char="═")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        if response.stop_reason == "max_tokens":
            print(f"⚠️  Warning: Model reached max_tokens limit ({MAX_TOKENS})")

        if verbose:
            print(f"\nStop reason: {response.stop_reason}")
            print(f"Usage: {response.usage.input_tokens} input, {response.usage.output_tokens} output tokens")

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose and content.text.strip():
                    print(f"\n💭 ASSISTANT THINKING:")
                    print("─" * 50)
                    text = content.text
                    # Show more of the assistant's reasoning
                    if len(text) > 1500:
                        print(text[:1500])
                        print(f"\n... (truncated {len(text) - 1500} chars)")
                    else:
                        print(text)

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                if tool_name in tool_handlers:
                    if verbose:
                        _log_tool_call(tool_name, tool_input)

                    handler = tool_handlers[tool_name]

                    if tool_name == "submit_answer":
                        # Handle malformed input (e.g., if model hit max_tokens)
                        if isinstance(tool_input, dict) and "answer" in tool_input:
                            answer = tool_input["answer"]
                        else:
                            answer = str(tool_input) if tool_input else ""
                            if verbose:
                                print(f"  ⚠️  Malformed submit input, using raw: {len(answer)} chars")
                        result = handler(answer)
                        submitted_answer = result["answer"]
                        if verbose:
                            print(f"  ✓ Answer submitted ({len(submitted_answer)} chars)")
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )
                        if verbose:
                            _log_tool_result(tool_name, result)

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
                    _print_header("SUBMITTED LITERATURE REVIEW", char="*")
                    print()
                    if len(submitted_answer) > 5000:
                        print(submitted_answer[:5000])
                        print(f"\n... (truncated {len(submitted_answer) - 5000} chars)")
                    else:
                        print(submitted_answer)
                    print()
                return submitted_answer
        else:
            if verbose:
                print("\n⏹️  No tool use in response, ending loop.")
            break

    if verbose:
        print(f"\n⚠️  Reached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    """Run a single test iteration."""
    if verbose:
        _print_header(f"RUN {run_id}/{num_runs}", char="█", width=80)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    result = await run_agent_loop(
        prompt=PROMPT,
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        max_steps=20,
        verbose=verbose,
    )

    if verbose:
        _print_header("GRADING PHASE", char="~", width=60)

    success = grading_func(result)

    display_result = result
    if isinstance(result, str) and len(result) > 200:
        display_result = result[:200] + "..."

    if success:
        print(f"\n✅ Run {run_id}: SUCCESS")
    else:
        print(f"\n❌ Run {run_id}: FAILURE")
        if not verbose:
            print(f"   Preview: {display_result}")

    return run_id, success, result


async def main(num_runs: int = 1, verbose: bool = True):
    """Run evaluation tests."""
    start_time = datetime.now()

    _print_header("LITERATURE REVIEW AGENT EVALUATION", char="█", width=80)
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Topic: {TOPIC:<67} ║
║  Model: claude-haiku-4-5 (research agent)                                    ║
║  Grader: claude-sonnet-4-5 (evaluation)                                      ║
║  Runs: {num_runs:<69} ║
║  Verbose: {str(verbose):<66} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

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
    elapsed = (datetime.now() - start_time).total_seconds()

    _print_header("FINAL RESULTS", char="█", width=80)
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EVALUATION COMPLETE                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✅ Passed: {successes}/{num_runs:<64} ║
║  ❌ Failed: {num_runs - successes}/{num_runs:<64} ║
║  📊 Pass Rate: {pass_rate:.1f}%{' ' * 60}║
║  ⏱️  Duration: {elapsed:.1f}s{' ' * 60}║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run literature review agent evaluation")
    parser.add_argument("-n", "--num-runs", type=int, default=1, help="Number of test runs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    asyncio.run(main(num_runs=args.num_runs, verbose=not args.quiet))
