"""
Idea Proposal Task Definition

This task:
1. Gives an agent access to literature reviews in RL, Memory, and Robotics
2. Asks it to generate novel research ideas prioritized by impact and business value
3. Uses a deep-thinking LLM-as-judge (Claude with extended thinking) to evaluate quality
"""

import os
from collections.abc import Callable
from typing import Any, TypedDict

import anthropic
from dotenv import load_dotenv

from .tools import TOOLS as RESEARCH_TOOLS, HANDLERS as RESEARCH_HANDLERS

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


class SubmitIdeasResult(TypedDict):
    ideas: Any
    submitted: bool


def submit_ideas_tool(ideas: str) -> SubmitIdeasResult:
    """Tool for submitting the final research ideas."""
    return {"ideas": ideas, "submitted": True}


# Submit tool definition
SUBMIT_TOOL = {
    "name": "submit_ideas",
    "description": "Submit your final research ideas proposal",
    "input_schema": {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "The complete research ideas document with prioritization"
            }
        },
        "required": ["ideas"],
    },
}

# Combined tools
TOOLS = RESEARCH_TOOLS + [SUBMIT_TOOL]

# Combined handlers
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    **RESEARCH_HANDLERS,
    "submit_ideas": submit_ideas_tool,
}

# The challenge prompt for idea generation
PROMPT = """You are a senior research scientist tasked with proposing novel, high-impact research ideas.

You have access to comprehensive literature reviews in three cutting-edge areas:
1. **Reinforcement Learning & Model Alignment** - RLHF, continual RL, world models, LLM+RL
2. **Long-Term Memory for LLMs** - context windows, RAG, memory architectures, model editing
3. **Data-Efficient Robotics** - foundation models, sim-to-real, imitation learning

## Your Task

1. First, use the tools to read the literature reviews (you can get all at once or individually)

2. Based on the current state of research, propose **5-7 novel research ideas** that:
   - Address REAL gaps or limitations identified in the literature
   - Could have HIGH IMPACT if successful (paradigm-shifting potential)
   - Have BUSINESS VALUE (practical applications, not just academic interest)
   - Are FEASIBLE within 1-2 years with current resources

3. For EACH idea, provide:
   - **Title**: A clear, descriptive name
   - **Problem**: What gap/limitation does this address?
   - **Approach**: High-level description of the proposed method
   - **Why Now**: Why is this tractable now but wasn't before?
   - **Impact Score** (1-10): How transformative if it works?
   - **Business Value** (1-10): Commercial/practical applications
   - **Feasibility** (1-10): Can this be done in 1-2 years?
   - **Key Risks**: What could go wrong?

4. **Prioritize your ideas** by a combined score: (Impact * 0.4) + (Business Value * 0.4) + (Feasibility * 0.2)
   Present them in order from highest to lowest priority.

## What Makes a GREAT Research Idea

- **Novel combination**: Connecting ideas across domains (e.g., memory techniques for robotics)
- **Contrarian insight**: Something the field hasn't considered or has dismissed
- **Clear evaluation**: A concrete way to measure success
- **Buildability**: Can start with a minimal viable experiment

## What to AVOID

- Incremental improvements ("make X slightly better")
- Ideas that are already being heavily pursued
- Pure engineering without scientific contribution
- Vague, hand-wavy concepts without concrete approaches

Submit your ideas using the submit_ideas tool when complete."""


# Deep-thinking evaluation rubric
JUDGE_SYSTEM_PROMPT = """You are a distinguished research director evaluating research proposals.

You have spent decades at the frontier of ML research and have seen countless proposals. You know:
- What makes ideas truly novel vs. rehashes of existing work
- The difference between incremental improvements and paradigm shifts
- How to spot ideas that sound good but won't work
- What actually has business value vs. academic curiosity

You will evaluate research ideas with brutal honesty. Good ideas are RARE - most ideas are mediocre or have fatal flaws. You should be skeptical and look for problems.

Your evaluation standards:
- **Novelty**: Is this genuinely new, or a minor variation on existing work?
- **Depth**: Does the proposer understand the problem deeply, or is this surface-level?
- **Feasibility**: Are there obvious technical blockers they haven't considered?
- **Impact**: Would this actually change anything if it worked?
- **Business Value**: Is there a real path to practical application?
- **Blind Spots**: What risks or challenges has the proposer missed?"""


JUDGE_EVALUATION_PROMPT = """## Research Ideas to Evaluate

{ideas}

## Your Task

Think deeply about these research ideas. Consider:

1. **For each idea**, assess:
   - Is it truly novel or a rehash?
   - Does the proposer understand the real challenges?
   - Are the claimed impact/value scores realistic or inflated?
   - What are the fatal flaws or blind spots?

2. **Overall assessment**:
   - How many of these ideas are actually good? (Most submissions have 0-2 truly good ideas)
   - Which idea is the BEST and why?
   - Which idea is the WORST and why?
   - What's missing that should have been proposed?

3. **Final Verdict**:
   - EXCEPTIONAL (4+ genuinely novel, high-impact ideas with realistic assessments)
   - GOOD (2-3 genuinely good ideas, shows real insight)
   - MEDIOCRE (1 decent idea, mostly obvious or flawed proposals)
   - POOR (no genuinely good ideas, surface-level thinking)

Be harsh but fair. Give specific reasons for your judgments.

## Response Format

For each idea:
```
IDEA: [Title]
NOVELTY: [1-10] - [One line explanation]
DEPTH: [1-10] - [One line explanation]
FEASIBILITY: [1-10] - [One line explanation]
REAL IMPACT: [1-10] - [One line explanation]
REAL BUSINESS VALUE: [1-10] - [One line explanation]
FATAL FLAWS: [List any dealbreakers]
VERDICT: [STRONG / PROMISING / MEDIOCRE / WEAK]
```

Then provide:
```
BEST IDEA: [Title] - [Why]
WORST IDEA: [Title] - [Why]
MISSING: [What should have been proposed]
OVERALL VERDICT: [EXCEPTIONAL / GOOD / MEDIOCRE / POOR]
```"""


def grading_func(result: Any) -> bool:
    """
    Evaluates research ideas using Claude with extended thinking as a deep judge.

    The judge thinks carefully about:
    - True novelty (not just incremental)
    - Realistic impact assessment
    - Feasibility and blind spots
    - Business value vs. academic interest

    Returns:
        True if the ideas are rated GOOD or EXCEPTIONAL, False otherwise
    """
    if not result or not isinstance(result, str):
        print("FAIL: No ideas submitted or not a string")
        return False

    ideas = result.strip()

    # Basic checks
    if len(ideas) < 500:
        print("FAIL: Ideas too short (< 500 characters)")
        return False

    # Check that multiple ideas were proposed
    idea_count = ideas.lower().count("idea") + ideas.lower().count("title:")
    if idea_count < 3:
        print("FAIL: Too few ideas proposed (need at least 3)")
        return False

    # Use Claude with extended thinking for deep evaluation
    client = anthropic.Anthropic()

    print("\n=== Research Ideas Evaluation (Deep Thinking Judge) ===")
    print("Sending ideas to evaluator with extended thinking...")

    try:
        # Use extended thinking for deep evaluation
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000  # Allow deep thinking
            },
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": JUDGE_EVALUATION_PROMPT.format(ideas=ideas)
            }]
        )

        # Extract the evaluation (non-thinking part)
        evaluation = ""
        thinking = ""
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                evaluation = block.text

        print("\n--- Judge's Thinking (Summary) ---")
        # Show first 1000 chars of thinking
        if thinking:
            print(thinking[:1500] + "..." if len(thinking) > 1500 else thinking)

        print("\n--- Evaluation ---")
        print(evaluation)

        # Parse the verdict
        eval_lower = evaluation.lower()

        if "overall verdict: exceptional" in eval_lower:
            print("\nPASS: Ideas rated EXCEPTIONAL")
            return True
        elif "overall verdict: good" in eval_lower:
            print("\nPASS: Ideas rated GOOD")
            return True
        elif "overall verdict: mediocre" in eval_lower:
            print("\nFAIL: Ideas rated MEDIOCRE")
            return False
        elif "overall verdict: poor" in eval_lower:
            print("\nFAIL: Ideas rated POOR")
            return False
        else:
            # Try to infer from context
            if "exceptional" in eval_lower or ("good" in eval_lower and "not good" not in eval_lower):
                print("\nPASS: Positive evaluation inferred")
                return True
            else:
                print("\nFAIL: Could not parse verdict, assuming negative")
                return False

    except Exception as e:
        print(f"FAIL: Error during evaluation: {e}")
        return False
