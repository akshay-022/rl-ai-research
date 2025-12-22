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
    TOOLS as RESEARCH_TOOLS,
    HANDLERS as RESEARCH_HANDLERS,
)

# ============================================================
# TASK DEFINITION (inline to avoid import issues)
# ============================================================

TOPIC = "giving LLMs long term memory"

PROMPT = f"""You are a research assistant conducting a literature review on: "{TOPIC}"

Your goal is to create a comprehensive literature review by:

1. Search for relevant papers using web_search with different queries covering all major approaches
2. For survey papers or highly-cited papers, use get_paper_introduction to extract the INTRODUCTION section
   - The introduction contains the authors' own literature review with context about related work
   - This is MUCH more valuable than abstracts - it shows how papers relate to each other
3. Use get_paper_with_tldr for quick summaries of individual papers
4. Use get_paper_references on 1-2 key papers to find foundational/seminal works
5. Synthesize everything into a structured literature review

IMPORTANT: For at least 2-3 key papers (especially survey papers), you MUST call get_paper_introduction
to extract the introduction section. The introduction contains the prior literature context that will
help you understand how different approaches relate to each other.

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

Use the available tools to gather information, then submit your complete literature review using submit_answer.
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
TOOLS = RESEARCH_TOOLS + [SUBMIT_TOOL]
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    **RESEARCH_HANDLERS,
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

## Reference Material (Ground Truth)
Below is a comprehensive reference covering the key topics and seminal works in this area:

{reference_material}

## Grading Criteria - BE STRICT

The review must demonstrate comprehensive coverage. Evaluate against these criteria:

### SECTION 1: Coverage of ALL Major Approaches (5 points - MUST GET ALL 5)

Each approach MUST be discussed with explanation AND at least one specific result/finding:

1. **Extended Context Windows** (1 point)
   - MUST mention: Long-context models (Claude 100k, GPT-4 128k, Gemini 1M), OR efficient attention (Longformer, BigBird, RMT)
   - MUST include a result: e.g., "Gemini 1.5 achieved 99% on needle-in-haystack at 1M tokens" or "RMT processed 2M tokens" or similar quantitative finding
   - FAIL if: Only mentions "longer context" without specifics or results

2. **Retrieval-Augmented Generation (RAG)** (1 point)
   - MUST mention: External knowledge stores, vector databases, retrieval systems
   - MUST mention at least ONE of: RETRO, RAG, Atlas, or similar retrieval-augmented systems
   - MUST include a result: e.g., "RETRO 7.5B outperformed 175B Jurassic-1" or retrieval improving accuracy
   - FAIL if: Just says "retrieval helps" without specific systems or results

3. **Summarization/Compression** (1 point)
   - MUST mention: Context compression, recursive summarization, conversation summaries, memory management
   - MUST include a finding: e.g., "recursive summarization improved long-dialogue consistency" or quantitative improvement
   - FAIL if: Only mentions "summarizing" without mechanism or results

4. **Specialized Memory Architectures** (1 point)
   - MUST mention at least ONE of: MemGPT, LongMem, key-value memory, hierarchical memory, memory graphs, Generative Agents
   - MUST explain the mechanism (e.g., "MemGPT treats LLM as OS managing memory")
   - MUST include a result: e.g., "LongMem achieved 40.5% on ChapterBreak" or similar
   - FAIL if: Only says "memory modules" without specific systems or results

5. **Parametric Memory / Weight Updates** (1 point)
   - MUST mention: Continual learning, fine-tuning for new knowledge, model editing (ROME, MEMIT), or fast weights
   - MUST discuss challenge: catastrophic forgetting
   - MUST include a finding: e.g., "MEMIT edited 5000 facts with minimal side effects"
   - FAIL if: Only mentions "training on new data" without specifics

### SECTION 2: Seminal Works with Results (3 points)

Award 1 point each ONLY if the paper is mentioned WITH a specific result:

1. **Retrieval Paper with Result** (1 point): RETRO, RAG, or Atlas mentioned with quantitative improvement
2. **Memory Architecture Paper with Result** (1 point): MemGPT, LongMem, or Generative Agents with specific finding
3. **Long Context Paper with Result** (1 point): Gemini, RMT, or efficient attention paper with benchmark result

### SECTION 3: Quality of Synthesis (2 points)

1. **Comparison/Trade-offs** (1 point): Explicitly compares approaches (e.g., retrieval vs parametric, context length vs efficiency, accuracy vs compute)
2. **Current Trends or Gaps** (1 point): Discusses recent trends (hybrid systems, ChatGPT memory) OR identifies research gaps

## Response Format

```
SECTION 1 - Major Approaches:
1. Extended Context: PASS/FAIL - [did they explain + give result?]
2. RAG/Retrieval: PASS/FAIL - [specific system + result?]
3. Summarization: PASS/FAIL - [mechanism + finding?]
4. Memory Architectures: PASS/FAIL - [specific system + result?]
5. Parametric Memory: PASS/FAIL - [discussed forgetting + result?]

SECTION 2 - Seminal Works with Results:
6. Retrieval Paper: PASS/FAIL - [paper name + quantitative result?]
7. Memory Paper: PASS/FAIL - [paper name + quantitative result?]
8. Long Context Paper: PASS/FAIL - [paper name + quantitative result?]

SECTION 3 - Synthesis:
9. Comparisons: PASS/FAIL - [explicit trade-off discussion?]
10. Trends/Gaps: PASS/FAIL - [current directions or gaps?]

TOTAL: X/10
RESULT: PASS (if >= 7/10) / FAIL (if < 7/10)
```

A submission PASSES only if it scores 7/10 or higher. Be STRICT - require specific papers and quantitative results."""


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
    elif tool_name == "get_paper_with_tldr":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
    elif tool_name == "get_paper_references":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
        print(f"  Limit: {tool_input.get('limit', 10)}")
        print(f"  (Will return abstracts and TLDRs for context)")
    elif tool_name == "get_paper_introduction":
        print(f"  Paper ID: {tool_input.get('paper_id', '')}")
        print(f"  (Extracting INTRODUCTION section from PDF - contains literature review)")
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

    elif tool_name == "get_paper_with_tldr":
        title = result.get("title", "Unknown")
        year = result.get("year", "?")
        citations = result.get("citation_count", "?")
        tldr = result.get("tldr")
        abstract = result.get("abstract")
        authors = result.get("authors", [])

        print(f"  ✓ Paper Details:")
        print(f"      Title: {title}")
        print(f"      Year: {year} | Citations: {citations}")
        if authors:
            print(f"      Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
        if tldr:
            print(f"\n      📝 TLDR:")
            print(_wrap_text(tldr, indent="         "))
        if abstract:
            short_abstract = abstract[:400] + "..." if len(abstract) > 400 else abstract
            print(f"\n      📄 Abstract (truncated):")
            print(_wrap_text(short_abstract, indent="         "))

    elif tool_name == "get_paper_references":
        refs = result.get("references", [])
        print(f"  ✓ Found {len(refs)} references WITH context:")
        print()
        for i, ref in enumerate(refs[:8], 1):  # Show up to 8 references
            title = ref.get("title", "Unknown")
            year = ref.get("year", "?")
            citations = ref.get("citation_count", "?")
            tldr = ref.get("tldr")
            abstract = ref.get("abstract")

            print(f"    [{i}] {title}")
            print(f"        Year: {year} | Citations: {citations}")

            # Show TLDR or abstract snippet for context
            if tldr:
                print(f"        📝 TLDR: {tldr[:150]}..." if len(tldr) > 150 else f"        📝 TLDR: {tldr}")
            elif abstract:
                short_abs = abstract[:150] + "..." if len(abstract) > 150 else abstract
                print(f"        📄 Abstract: {short_abs}")
            print()
        if len(refs) > 8:
            print(f"    ... and {len(refs) - 8} more references")

    elif tool_name == "get_paper_introduction":
        title = result.get("title", "Unknown")
        arxiv_id = result.get("arxiv_id", "")
        introduction = result.get("introduction")
        abstract = result.get("abstract")

        print(f"  ✓ Paper Introduction Extracted:")
        print(f"      Title: {title}")
        if arxiv_id:
            print(f"      arXiv ID: {arxiv_id}")
        if introduction:
            print(f"\n      📖 INTRODUCTION (literature review context):")
            print(f"      {'─' * 60}")
            # Show more of the introduction since it's valuable
            if len(introduction) > 2000:
                print(_wrap_text(introduction[:2000], indent="         "))
                print(f"\n         ... ({len(introduction) - 2000} more chars)")
            else:
                print(_wrap_text(introduction, indent="         "))
        elif abstract:
            print(f"\n      ⚠️  Could not extract introduction, showing abstract:")
            print(_wrap_text(abstract, indent="         "))

    else:
        # Generic result display
        print(f"  {json.dumps(result, indent=2)[:800]}")


# ============================================================
# AGENT LOOP
# ============================================================

MAX_TOKENS = 8000  # Increased for longer responses


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
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
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
