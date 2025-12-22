"""
Progressive Titans Evaluation Suite

Tests an agent's ability to "discover" the Titans architecture step-by-step.
Each step is INDEPENDENT and runs in PARALLEL - no cascading dependencies.

For later steps that need conceptual context, brief Titans paper hints are provided.
"""

import os
import concurrent.futures
from dotenv import load_dotenv
import anthropic

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


# --- THE 5-STEP PROGRESSIVE PROMPTS ---
# Each prompt is SELF-CONTAINED and requires genuine reasoning
# No paper references - agent must derive the architecture independently

STEP_1_PROMPT = """I want to add a memory system to a Transformer that can handle very long sequences without the quadratic attention cost.

What kind of memory structure would you propose? Describe its mathematical form."""

STEP_2_PROMPT = """Consider a neural memory system where we store information in a weight matrix M that gets updated for every token.

If you update this memory for every token, won't it get overwhelmed with noise?

What's the problem here and how would you address it?"""

STEP_3_PROMPT = """Consider a neural memory system for Transformers where:
- Memory is stored as a matrix M
- We need to control which tokens get written to memory

Give me the complete mathematical update rule for this memory.

Write it as a single equation: M_new = ..."""

STEP_4_PROMPT = """Consider a Transformer block that has BOTH:
1. Standard attention (for short-term context)
2. A neural memory module (for long-term storage)

How do you combine their outputs in a single block?"""

STEP_5_PROMPT = """Implement a neural memory module in PyTorch that:
- Uses a matrix M to store information
- Updates M recurrently as it processes each token
- Can be combined with standard attention in a Transformer block

Give me complete, runnable code for a `NeuralMemory` module and a `MemoryBlock` that uses both attention and memory."""


PROMPTS = [
    ("Step 1: High-Dimensional Memory", STEP_1_PROMPT),
    ("Step 2: Surprise/Importance Filter", STEP_2_PROMPT),
    ("Step 3: Update Rule with Decay", STEP_3_PROMPT),
    ("Step 4: Attention-Memory Fusion", STEP_4_PROMPT),
    ("Step 5: Implementation", STEP_5_PROMPT),
]


# --- GRADING RUBRICS FOR EACH STEP ---
# These rubrics are STRICT - we want Haiku to fail 60-90% of the time

STEP_1_RUBRIC = """Evaluate if the response proposes a HIGH-DIMENSIONAL memory structure that uses OUTER PRODUCTS.

**PASS criteria (must meet BOTH):**
1. Proposes memory as a matrix M with dimensions like (d x d) - NOT just (memory_slots x d)
2. Explicitly mentions using "outer product" (v ⊗ k^T) to store key-value associations

**FAIL criteria (any of these = FAIL):**
- Proposes memory as (n x d) where n is just number of slots - this is just a list of vectors
- Uses standard attention over memory slots instead of outer product
- Proposes retrieval-based memory (like memory networks) instead of fast weights
- No explicit outer product formulation

Be STRICT. A memory bank of shape (k x d) is NOT the same as a fast-weight matrix (d x d).

Respond with exactly:
STEP 1: PASS or FAIL
Reason: [brief explanation]"""

STEP_2_RUBRIC = """Evaluate if the response proposes a SURPRISE-BASED filtering mechanism.

**PASS criteria (must meet BOTH):**
1. Proposes measuring "surprise" or "novelty" - how unexpected the input is
2. This surprise metric is used to GATE memory updates (not just filter/select)

**FAIL criteria (any of these = FAIL):**
- Only proposes a generic "importance" or "relevance" gate without surprise/novelty
- Uses attention scores as the gate (this is relevance, not surprise)
- Gate is based on position or frequency, not content unpredictability
- No mechanism that measures how "surprising" or "novel" the input is

The key insight is: surprise = "how wrong was my prediction?" NOT "how relevant is this?"

Respond with exactly:
STEP 2: PASS or FAIL
Reason: [brief explanation]"""

STEP_3_RUBRIC = """Evaluate if the response proposes the EXACT update rule with BOTH decay AND surprise.

**PASS criteria (must meet ALL THREE):**
1. Has explicit DECAY term: M_new = decay * M_old + ... (where decay < 1)
2. Has explicit SURPRISE term that gates the update magnitude
3. Uses OUTER PRODUCT for the update: decay * M + surprise * (V ⊗ K^T)

**FAIL criteria (any of these = FAIL):**
- Missing decay term (just M_new = M + update)
- Missing surprise gating (just M_new = decay*M + update)
- No outer product (V ⊗ K^T) in the formula
- Uses interpolation like M = (1-W)*M + W*V without outer product
- Has decay OR surprise but not BOTH

This is strict: the formula MUST be: M_new = decay * M_old + surprise * (V ⊗ K^T)
Or equivalent with both components clearly present.

Respond with exactly:
STEP 3: PASS or FAIL
Reason: [brief explanation]"""

STEP_4_RUBRIC = """Evaluate if the response proposes LEARNED SIGMOID GATING for fusion.

**PASS criteria (must meet ALL):**
1. Uses sigmoid activation specifically (not softmax, not learned weights alone)
2. Computes gate from BOTH attention and memory outputs: g = sigmoid(W[attn; mem])
3. Final output uses complementary weighting: out = g*attn + (1-g)*mem

**FAIL criteria (any of these = FAIL):**
- Uses simple addition or concatenation
- Uses only attention or only memory to compute the gate (must use both)
- Uses softmax instead of sigmoid
- Uses fixed/hardcoded mixing weights
- Missing the (1-g) complementary term

The exact pattern required: g = sigmoid(linear([attn, mem])), then g*attn + (1-g)*mem

Respond with exactly:
STEP 4: PASS or FAIL
Reason: [brief explanation]"""

STEP_5_RUBRIC = """Evaluate the FINAL IMPLEMENTATION code. Be VERY STRICT.

**Must have ALL 6 components to PASS:**

1. **NeuralMemory class** - Proper nn.Module with forward method
2. **Surprise gate** - Uses sigmoid to compute surprise from prediction error or learned gate
3. **Learnable decay** - nn.Parameter for memory fade (not hardcoded)
4. **Outer product** - V⊗K^T computed via torch.bmm or einsum (not matmul on batched vectors)
5. **Recurrent loop** - Explicit for loop: "for t in range(seq_len)" updating state each step
6. **Fusion gate** - g = sigmoid(...) then g*attn + (1-g)*mem pattern

**FAIL if missing ANY of these.** Partial credit does not count as pass.

Respond with exactly:
STEP 5: PASS or FAIL
Components passed: X/6
Reason: [brief explanation]"""

RUBRICS = [
    STEP_1_RUBRIC,
    STEP_2_RUBRIC,
    STEP_3_RUBRIC,
    STEP_4_RUBRIC,
    STEP_5_RUBRIC,
]


def _run_single_step(client, model: str, step_idx: int, step_name: str, prompt: str, rubric: str):
    """
    Run a single step of the evaluation (agent call + grading).
    This is called in parallel for all 5 steps.
    """
    # Get response from model being tested (independent call, no conversation history)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    agent_response = response.content[0].text

    # Grade this step using Sonnet
    grade_prompt = f"""Grade the following response against the rubric.

RESPONSE TO GRADE:
{agent_response}

RUBRIC:
{rubric}"""

    grade_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": grade_prompt
        }]
    )

    grade_text = grade_response.content[0].text

    # Parse result
    passed = "pass" in grade_text.lower().split("\n")[0]

    return {
        "step": step_idx + 1,
        "name": step_name,
        "passed": passed,
        "response": agent_response,
        "grade": grade_text,
        "prompt": prompt,
    }


def run_progressive_evaluation(model: str = "claude-haiku-4-5"):
    """
    Run the 5-step progressive evaluation on a model IN PARALLEL.

    Each step is independent - no cascading dependencies between steps.

    Args:
        model: The model to test (e.g., "claude-haiku-4-5")

    Returns:
        dict with results for each step
    """
    client = anthropic.Anthropic()

    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TITANS EVALUATION (PARALLEL)")
    print(f"Model: {model}")
    print(f"{'='*60}")
    print(f"\nRunning all 5 steps in parallel...")

    # Run all 5 steps in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, (step_name, prompt) in enumerate(PROMPTS):
            future = executor.submit(
                _run_single_step,
                client, model, i, step_name, prompt, RUBRICS[i]
            )
            futures.append(future)

        # Collect results as they complete
        step_results = [f.result() for f in futures]

    # Sort by step number to ensure correct order
    step_results.sort(key=lambda x: x["step"])

    # Print results
    for step in step_results:
        print(f"\n{'='*60}")
        print(f"{step['name']}")
        print(f"{'='*60}")
        print(f"\n--- Prompt ---")
        prompt_text = step['prompt']
        print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
        print(f"\n--- {model} Response ---")
        response_text = step['response']
        print(response_text[:1500] + "..." if len(response_text) > 1500 else response_text)
        print(f"\n--- Grade ---")
        print(step['grade'])

    # Compute totals
    total_passed = sum(1 for s in step_results if s["passed"])

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    for step in step_results:
        status = "✓ PASS" if step["passed"] else "✗ FAIL"
        print(f"Step {step['step']}: {step['name']} - {status}")

    print(f"\nTotal: {total_passed}/5 steps passed")

    # Overall pass requires at least 4/5 steps
    overall_pass = total_passed >= 4
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'} (need 4/5)")

    results = {
        "model": model,
        "steps": step_results,
        "total_passed": total_passed,
        "overall_pass": overall_pass,
    }

    return results


def run_comparison(models: list[str] = None):
    """
    Run the evaluation on multiple models and compare results.
    """
    if models is None:
        models = [
            "claude-haiku-4-5",
            # Add other models to compare
        ]

    all_results = {}

    for model in models:
        print(f"\n\n{'#'*60}")
        print(f"# TESTING: {model}")
        print(f"{'#'*60}")

        results = run_progressive_evaluation(model)
        all_results[model] = results

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    for model, results in all_results.items():
        steps_passed = [s["passed"] for s in results["steps"]]
        step_str = "".join(["✓" if p else "✗" for p in steps_passed])
        print(f"{model}: {step_str} ({results['total_passed']}/5) - {'PASS' if results['overall_pass'] else 'FAIL'}")

    return all_results


if __name__ == "__main__":
    # Run on Haiku to test
    results = run_progressive_evaluation("claude-haiku-4-5")
