"""
Literature Review Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
"""

from collections.abc import Callable
from typing import Any, TypedDict

from .tools import TOOLS as RESEARCH_TOOLS, HANDLERS as RESEARCH_HANDLERS


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

1. Search for the most relevant recent papers on this topic (use web_search)
2. For the top 3-5 papers, get their detailed summaries (use get_paper_with_tldr)
3. For the most relevant paper, explore its references to find foundational work (use get_paper_references)
4. Synthesize everything into a structured literature review

Your final review should cover:
- Main approaches/methods being used in this area
- Key results and findings from important papers
- How different papers relate to each other
- Current trends and potential gaps in the research

Use the available tools to gather information, then submit your complete literature review using submit_answer.
Be thorough but concise. Cite specific papers by title when discussing their contributions."""


def grading_func(result: Any) -> bool:
    """
    Validates the literature review.

    Checks that the review:
    1. Is a non-empty string of reasonable length
    2. Mentions memory-related concepts
    3. Discusses methods/approaches
    4. Includes findings/results

    Returns:
        True if the review meets quality criteria, False otherwise
    """
    if not result or not isinstance(result, str):
        return False

    review = result.lower()

    # Must be substantial (at least 500 chars)
    if len(result) < 500:
        return False

    # Should mention memory-related concepts
    memory_terms = ["memory", "long-term", "retrieval", "context", "storage"]
    has_memory_discussion = any(term in review for term in memory_terms)

    # Should discuss methods/approaches
    method_terms = ["method", "approach", "technique", "architecture", "model", "framework"]
    has_methods = any(term in review for term in method_terms)

    # Should mention results/findings
    result_terms = ["result", "finding", "show", "demonstrate", "achieve", "performance", "improve"]
    has_results = any(term in review for term in result_terms)

    return has_memory_discussion and has_methods and has_results
