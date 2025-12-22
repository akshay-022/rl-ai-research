"""
Literature Review Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer using LLM grading
"""

import os
from collections.abc import Callable
from typing import Any, TypedDict

import anthropic
from dotenv import load_dotenv

from .tools import TOOLS as RESEARCH_TOOLS, HANDLERS as RESEARCH_HANDLERS

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Tool for submitting the final literature review."""
    return {"answer": answer, "submitted": True}


# Submit tool definition
SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final literature review",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "The complete literature review"}},
        "required": ["answer"],
    },
}

# Combined tools
TOOLS = RESEARCH_TOOLS + [SUBMIT_TOOL]

# Combined handlers
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    **RESEARCH_HANDLERS,
    "submit_answer": submit_answer_tool,
}

# The research topic for the literature review
TOPIC = "giving LLMs long term memory"

# The challenge prompt
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


# Load reference material for grading
def _load_reference_material():
    """Load the reference literature review from the dataset."""
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    memory_path = os.path.join(dataset_dir, 'memory.md')

    if os.path.exists(memory_path):
        with open(memory_path, 'r') as f:
            return f.read()
    return None


REFERENCE_MATERIAL = _load_reference_material()


# Grading rubric for LLM evaluation - STRICT version requiring ALL topics and results
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

    The review is checked against the reference material to ensure it covers:
    - ALL major approaches to LLM memory (with results)
    - Seminal works with quantitative findings
    - Quality synthesis and comparison

    Returns:
        True if the review scores 7/10 or higher, False otherwise
    """
    if not result or not isinstance(result, str):
        print("FAIL: No review submitted or not a string")
        return False

    review = result.strip()

    # Basic length check - comprehensive review needs to be substantial
    if len(review) < 1000:
        print("FAIL: Review too short (< 1000 characters) for comprehensive coverage")
        return False

    # If no reference material, fall back to basic checks
    if not REFERENCE_MATERIAL:
        print("WARNING: No reference material found, using basic grading")
        review_lower = review.lower()
        memory_terms = ["memory", "long-term", "retrieval", "context", "storage"]
        method_terms = ["method", "approach", "technique", "architecture", "model"]
        has_memory = any(term in review_lower for term in memory_terms)
        has_methods = any(term in review_lower for term in method_terms)
        return has_memory and has_methods

    # Use Claude Sonnet to evaluate against reference material
    client = anthropic.Anthropic()

    # Truncate reference material if too long (keep key sections)
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

        # Parse the result
        eval_lower = evaluation.lower()

        # Count passes
        pass_count = eval_lower.count(": pass")

        # Check for final result - need 7/10 to pass
        if "result: pass" in eval_lower and pass_count >= 7:
            print(f"\nPASS: {pass_count}/10 criteria passed")
            return True
        else:
            print(f"\nFAIL: Only {pass_count}/10 criteria passed (need 7/10)")
            return False

    except Exception as e:
        print(f"FAIL: Error during evaluation: {e}")
        return False
