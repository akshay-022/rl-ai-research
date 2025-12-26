"""
Test script for the idea-proposal task.
Uses Haiku agent loop for idea generation, Sonnet extended thinking for evaluation.

Usage:
  python test_task.py              # Full mode: all 3 literatures
  python test_task.py --memory     # Quick: just memory literature
  python test_task.py --rl         # Quick: just RL literature
  python test_task.py --robotics   # Quick: just robotics literature
"""

import asyncio
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anthropic import AsyncAnthropic, Anthropic
import importlib

# Import literature dataset tools
idea_proposal_tools = importlib.import_module("tasks.idea-proposal.tools")
DATASET_TOOLS = idea_proposal_tools.TOOLS
DATASET_HANDLERS = idea_proposal_tools.HANDLERS

# Import paper research tools (search, TLDR, references, etc.)
lit_review_tools = importlib.import_module("tasks.literature-review.tools")
PAPER_TOOLS = lit_review_tools.TOOLS
PAPER_HANDLERS = lit_review_tools.HANDLERS

# Combine all tools
ALL_TOOLS = DATASET_TOOLS + PAPER_TOOLS

# Submit tool for when Haiku is done
SUBMIT_TOOL = {
    "name": "submit_ideas",
    "description": "Submit your final research ideas when you're done. Call this once you have formulated your ideas.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "Your complete research ideas document"
            }
        },
        "required": ["ideas"],
    },
}

ALL_TOOLS = ALL_TOOLS + [SUBMIT_TOOL]

# Combined handlers
ALL_HANDLERS = {
    **DATASET_HANDLERS,
    **PAPER_HANDLERS,
    "submit_ideas": lambda ideas: {"ideas": ideas, "submitted": True},
}

# Prompt for Haiku agent (full mode)
GENERATION_PROMPT = """You are a research scientist tasked with proposing novel, high-impact research ideas.

## Available Tools

You have access to:
1. **Literature datasets** - Comprehensive reviews on RL, Memory, and Robotics
   - get_all_literature() - Get all three reviews at once
   - get_rl_literature(), get_memory_literature(), get_robotics_literature() - Individual reviews

2. **Paper research tools** - For deeper investigation
   - web_search(query) - Search for papers on arXiv/Semantic Scholar
   - get_paper_with_tldr(paper_id) - Get paper details and AI summary
   - get_paper_references(paper_id) - Get papers that a paper cites
   - get_paper_results(paper_id) - Extract results/experiments section

3. **submit_ideas(ideas)** - Submit your final ideas when done

## Your Task

1. Start by reading the literature reviews to understand the landscape
2. Identify gaps, limitations, and opportunities
3. Optionally search for specific papers to validate your ideas or find supporting evidence
4. Propose 5 novel research ideas that:
   - Address REAL gaps identified in the literature
   - Connect concepts ACROSS domains (RL + Memory + Robotics)
   - Have clear business/practical value
   - Are feasible in 1-2 years

## For Each Idea, Provide:

- **Title**: Clear, descriptive name
- **Problem**: What gap/limitation does this address? (cite specific papers/findings)
- **Approach**: High-level method description
- **Why Now**: Why is this tractable now? What recent advances enable it?
- **Expected Outcome**: What would success look like? What metrics?
- **Business Value**: Who would pay for this? What's the market?

## When Done

Call submit_ideas with your complete proposal. Take your time - research thoroughly before proposing."""


async def generate_ideas_with_haiku() -> str:
    """Use Haiku in continuous agent loop until it submits ideas."""
    client = AsyncAnthropic()
    messages = [{"role": "user", "content": GENERATION_PROMPT}]

    step = 0
    max_steps = 30  # Safety limit, but expect ~5-10 steps

    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")

        start = time.time()
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8000,
            tools=ALL_TOOLS,
            messages=messages
        )
        elapsed = time.time() - start
        print(f"API call: {elapsed:.1f}s | Stop reason: {response.stop_reason}")

        has_tool_use = False
        tool_results = []
        submitted_ideas = None

        for content in response.content:
            if content.type == "text":
                text = content.text
                if len(text) > 500:
                    print(f"\nHaiku: {text[:500]}...")
                elif text.strip():
                    print(f"\nHaiku: {text}")

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                print(f"\n>>> Tool: {tool_name}", end="")
                if tool_input:
                    # Show relevant input params
                    if "query" in tool_input:
                        print(f" | query='{tool_input['query']}'", end="")
                    if "paper_id" in tool_input:
                        print(f" | paper_id='{tool_input['paper_id']}'", end="")
                print()

                if tool_name == "submit_ideas":
                    submitted_ideas = tool_input["ideas"]
                    print(f"\n{'='*60}")
                    print("IDEAS SUBMITTED!")
                    print(f"{'='*60}")
                    result = {"ideas": submitted_ideas, "submitted": True}
                elif tool_name in ALL_HANDLERS:
                    handler = ALL_HANDLERS[tool_name]
                    try:
                        # Call with appropriate args
                        if tool_input:
                            result = handler(**tool_input)
                        else:
                            result = handler()

                        # Show result summary
                        if isinstance(result, dict):
                            if "topics" in result:
                                for t in result["topics"]:
                                    print(f"    - {t['name']}: {len(t['content'])} chars")
                            elif "topic" in result:
                                print(f"    - {result['topic']}: {len(result.get('content', ''))} chars")
                            elif "results" in result and isinstance(result["results"], list):
                                print(f"    Found {len(result['results'])} papers")
                                for p in result["results"][:3]:
                                    print(f"      - {p.get('title', 'Unknown')[:60]}...")
                            elif "title" in result:
                                print(f"    - {result['title'][:60]}...")
                                if result.get("tldr"):
                                    print(f"    - TLDR: {result['tldr'][:100]}...")
                            elif "error" in result and result["error"]:
                                print(f"    ERROR: {result['error']}")
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": json.dumps(result),
                })

        # Check if ideas were submitted
        if submitted_ideas is not None:
            return submitted_ideas

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool use and no submission - might be done thinking
            print("\nNo tool use in response. Prompting to submit...")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "Please submit your ideas using the submit_ideas tool."})

    print(f"\nReached max steps ({max_steps}) without submission")
    return ""


def evaluate_ideas_with_thinking(ideas: str) -> dict:
    """Use Sonnet with extended thinking to deeply evaluate the ideas."""
    client = Anthropic()

    eval_prompt = f"""## Research Ideas to Evaluate

{ideas}

## Evaluation Criteria

For each idea, assess based on EVIDENCE and PRIOR WORK:

1. **Business Viability (1-10)**:
   - Is there a clear path to commercial application?
   - Who specifically would pay for this? (name companies/industries)
   - What's the market size?

2. **Potential Upside (1-10)**:
   - If this works, how transformative is it?
   - 10 = paradigm shift (like transformers, RLHF), 1 = incremental improvement

3. **Likelihood of Success (1-10)**:
   - Based on PRIOR WORK cited, how likely is this to work?
   - Have similar approaches succeeded before?
   - Are the prerequisites (data, compute, algorithms) in place?
   - What's the track record of the underlying techniques?

4. **Novelty (1-10)**:
   - Is this genuinely new or a straightforward combination?
   - Has this been tried before? Why did/didn't it work?

5. **Key Risks**: What could kill this idea? Technical blockers? Market issues?

## Output Format

For each idea:
```
IDEA: [Title]
Business Viability: X/10 - [specific customers/market]
Potential Upside: X/10 - [comparison to known breakthroughs]
Likelihood of Success: X/10 - [cite prior work supporting/against]
Novelty: X/10 - [what's actually new here]
Key Risks: [list concrete risks]
OVERALL: [STRONG / PROMISING / WEAK]
```

Then:
```
RANKING: [ordered from best to worst]
BEST IDEA: [which and why - be specific]
WORST IDEA: [which and why - be specific]
VERDICT: [EXCEPTIONAL / GOOD / MEDIOCRE / POOR]
```

Be brutally honest. Cite specific prior work when assessing likelihood of success.

## Scoring

For each idea, compute an OVERALL SCORE (1-10) as:
  Score = (Business Viability * 0.3) + (Potential Upside * 0.25) + (Likelihood of Success * 0.3) + (Novelty * 0.15)

At the end, provide:
```
SCORES:
- Idea 1: X.X/10
- Idea 2: X.X/10
- Idea 3: X.X/10
...
AVERAGE SCORE: X.X/10
```"""

    print("\nCalling Sonnet with extended thinking (10k budget)...")
    start = time.time()

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{"role": "user", "content": eval_prompt}]
    )

    print(f"Thinking call took: {time.time() - start:.1f}s")

    thinking = ""
    evaluation = ""
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            evaluation = block.text

    # Parse average score from evaluation
    import re
    avg_match = re.search(r'AVERAGE SCORE:\s*(\d+\.?\d*)/10', evaluation)
    avg_score = float(avg_match.group(1)) if avg_match else None

    return {
        "thinking": thinking,
        "evaluation": evaluation,
        "average_score": avg_score
    }


# Topic-specific prompts
TOPIC_PROMPTS = {
    "memory": {
        "name": "LLM Long-Term Memory",
        "tool": "get_memory_literature",
        "prompt": """You are a research scientist proposing novel research ideas on LLM memory.

## Available Tools

1. **get_memory_literature()** - Get the literature review on LLM long-term memory
2. **Paper research tools** (optional):
   - web_search(query) - Search for papers
   - get_paper_with_tldr(paper_id) - Get paper details
   - get_paper_results(paper_id) - Get results section
3. **submit_ideas(ideas)** - Submit your final ideas

## Task

1. Call get_memory_literature() to read the review
2. Propose 3 novel research ideas with:
   - **Title**: Clear name
   - **Problem**: What gap does this address?
   - **Approach**: How would you solve it?
   - **Business Value**: Who would pay for this?

Call submit_ideas when done."""
    },
    "rl": {
        "name": "Reinforcement Learning & Alignment",
        "tool": "get_rl_literature",
        "prompt": """You are a research scientist proposing novel research ideas on RL and model alignment.

## Available Tools

1. **get_rl_literature()** - Get the literature review on RL and alignment
2. **Paper research tools** (optional):
   - web_search(query) - Search for papers
   - get_paper_with_tldr(paper_id) - Get paper details
   - get_paper_results(paper_id) - Get results section
3. **submit_ideas(ideas)** - Submit your final ideas

## Task

1. Call get_rl_literature() to read the review
2. Propose 3 novel research ideas with:
   - **Title**: Clear name
   - **Problem**: What gap does this address?
   - **Approach**: How would you solve it?
   - **Business Value**: Who would pay for this?

Call submit_ideas when done."""
    },
    "robotics": {
        "name": "Data-Efficient Robotics",
        "tool": "get_robotics_literature",
        "prompt": """You are a research scientist proposing novel research ideas on robotics learning.

## Available Tools

1. **get_robotics_literature()** - Get the literature review on data-efficient robotics
2. **Paper research tools** (optional):
   - web_search(query) - Search for papers
   - get_paper_with_tldr(paper_id) - Get paper details
   - get_paper_results(paper_id) - Get results section
3. **submit_ideas(ideas)** - Submit your final ideas

## Task

1. Call get_robotics_literature() to read the review
2. Propose 3 novel research ideas with:
   - **Title**: Clear name
   - **Problem**: What gap does this address?
   - **Approach**: How would you solve it?
   - **Business Value**: Who would pay for this?

Call submit_ideas when done."""
    },
}


async def generate_ideas_for_topic(topic: str) -> str:
    """Quick mode: agent loop but focused on one topic only."""
    topic_config = TOPIC_PROMPTS[topic]
    print(f"Topic: {topic_config['name']}")

    client = AsyncAnthropic()
    messages = [{"role": "user", "content": topic_config["prompt"]}]

    step = 0
    max_steps = 15  # Fewer steps needed for single topic

    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")

        start = time.time()
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4000,
            tools=ALL_TOOLS,
            messages=messages
        )
        elapsed = time.time() - start
        print(f"API call: {elapsed:.1f}s | Stop reason: {response.stop_reason}")

        has_tool_use = False
        tool_results = []
        submitted_ideas = None

        for content in response.content:
            if content.type == "text":
                text = content.text
                if len(text) > 500:
                    print(f"\nHaiku: {text[:500]}...")
                elif text.strip():
                    print(f"\nHaiku: {text}")

            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input

                print(f"\n>>> Tool: {tool_name}", end="")
                if tool_input:
                    if "query" in tool_input:
                        print(f" | query='{tool_input['query']}'", end="")
                    if "paper_id" in tool_input:
                        print(f" | paper_id='{tool_input['paper_id']}'", end="")
                print()

                if tool_name == "submit_ideas":
                    submitted_ideas = tool_input["ideas"]
                    print(f"\n{'='*60}")
                    print("IDEAS SUBMITTED!")
                    print(f"{'='*60}")
                    result = {"ideas": submitted_ideas, "submitted": True}
                elif tool_name in ALL_HANDLERS:
                    handler = ALL_HANDLERS[tool_name]
                    try:
                        if tool_input:
                            result = handler(**tool_input)
                        else:
                            result = handler()

                        if isinstance(result, dict):
                            if "topic" in result:
                                print(f"    - {result['topic']}: {len(result.get('content', ''))} chars")
                            elif "results" in result and isinstance(result["results"], list):
                                print(f"    Found {len(result['results'])} papers")
                            elif "title" in result:
                                print(f"    - {result['title'][:60]}...")
                            elif "error" in result and result["error"]:
                                print(f"    ERROR: {result['error']}")
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": json.dumps(result),
                })

        if submitted_ideas is not None:
            return submitted_ideas

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            print("\nNo tool use. Prompting to submit...")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "Please submit your ideas using submit_ideas tool."})

    print(f"\nReached max steps ({max_steps})")
    return ""


async def main(topic: str | None = None):
    print("="*60)
    if topic:
        print(f"IDEA PROPOSAL TASK ({topic.upper()} ONLY)")
    else:
        print("IDEA PROPOSAL TASK (FULL MODE - ALL TOPICS)")
    print("Haiku Agent -> Sonnet Extended Thinking Judge")
    print("="*60)

    # Phase 1: Generate ideas
    print("\n\nPHASE 1: GENERATING IDEAS")
    print("-"*60)
    start_gen = time.time()

    if topic:
        ideas = await generate_ideas_for_topic(topic)
    else:
        ideas = await generate_ideas_with_haiku()

    gen_time = time.time() - start_gen

    if not ideas or len(ideas) < 100:
        print("\nFAILED: No substantial ideas generated")
        return

    print(f"\n\n{'='*60}")
    print(f"GENERATED IDEAS ({len(ideas)} chars, took {gen_time:.1f}s)")
    print("="*60)
    print(ideas)
    print("="*60)

    # Phase 2: Evaluate with Sonnet thinking
    print("\n\nPHASE 2: SONNET DEEP EVALUATION")
    print("-"*60)
    result = evaluate_ideas_with_thinking(ideas)

    print("\n--- THINKING (first 2500 chars) ---")
    thinking_preview = result["thinking"][:2500]
    if len(result["thinking"]) > 2500:
        thinking_preview += "\n... [truncated]"
    print(thinking_preview)

    print("\n\n--- EVALUATION ---")
    print(result["evaluation"])

    # Show score and verdict
    avg_score = result.get("average_score")
    verdict = result["evaluation"].lower()

    print("\n" + "="*60)
    if avg_score:
        print(f"AVERAGE SCORE: {avg_score}/10")

    if "verdict: exceptional" in verdict or "verdict: good" in verdict:
        print("FINAL RESULT: PASS")
    else:
        print("FINAL RESULT: FAIL")
    print("="*60)


if __name__ == "__main__":
    # Parse topic from args
    topic = None
    if "--memory" in sys.argv:
        topic = "memory"
    elif "--rl" in sys.argv:
        topic = "rl"
    elif "--robotics" in sys.argv:
        topic = "robotics"

    asyncio.run(main(topic=topic))
