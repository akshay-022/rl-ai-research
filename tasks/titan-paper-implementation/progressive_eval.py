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

What kind of memory structure would you propose? Describe its mathematical form.

Also provide pseudocode showing how this memory would be initialized and updated."""

STEP_2_PROMPT = """Consider a neural memory system where we store information in a weight matrix M that gets updated for every token.

If you update this memory for every token, won't it get overwhelmed with noise?

What's the problem here and how would you address it?

Provide pseudocode showing your proposed filtering mechanism."""

STEP_3_PROMPT = """Consider a neural memory system for Transformers where:
- Memory is stored as a matrix M
- We need to control which tokens get written to memory

Give me the complete mathematical update rule for this memory.

Write it as a single equation: M_new = ...

Also provide pseudocode showing how this update rule would be applied step by step."""

STEP_4_PROMPT = """Consider a Transformer block that has BOTH:
1. Standard attention (for short-term context)
2. A neural memory module (for long-term storage)

How should you combine their outputs? Just adding them (attn + mem) seems too simple.

What's a better approach and why?

Provide pseudocode showing how the fusion would work."""

STEP_5_PROMPT = """Implement a neural memory module in PyTorch.

Requirements:
- Store information in a matrix M (not a list of vectors)
- Update M for each token using outer products
- Include a mechanism to prevent memory saturation
- Process tokens one at a time (recurrently)

Give me complete, runnable code for a `NeuralMemory` class."""


PROMPTS = [
    ("Step 1: High-Dimensional Memory", STEP_1_PROMPT),
    ("Step 2: Surprise/Importance Filter", STEP_2_PROMPT),
    ("Step 3: Update Rule with Decay", STEP_3_PROMPT),
    ("Step 4: Attention-Memory Fusion", STEP_4_PROMPT),
    ("Step 5: Implementation", STEP_5_PROMPT),
]


# --- GRADING RUBRICS FOR EACH STEP ---
# Philosophy: Be STRICT. We want only 10-40% pass rate (60-90% fail rate)
# The agent must demonstrate deep understanding, not just surface-level answers

STEP_1_RUBRIC = """You are evaluating if the response proposes ASSOCIATIVE/MATRIX memory (not simple vector storage).

**PASS if ANY of these appear (be generous):**
- Symbols: ⊗, outer(x,y), v @ k^T, v k^T, rank-1
- Terms: "outer product", "fast weights", "weight matrix", "associative memory"
- Terms: "Hopfield", "attractor", "energy-based"
- Linear attention with kernel/feature maps (implicit associative)
- ANY matrix M that gets UPDATED additively: M ← M + ...
- Memory described as (d × d) or (d_k × d_v) matrix shape

**FAIL only if:**
- Memory is ONLY described as list/buffer of vectors
- ONLY proposes K,V cache with attention retrieval
- No matrix update rule at all

Be GENEROUS. If there's ANY matrix update equation (M = M + ...), pass it.
The response doesn't need perfect outer product notation - just matrix-based memory.

Respond with exactly:
STEP 1: PASS or FAIL
Reason: [Quote the matrix update or associative term found.]"""

STEP_2_RUBRIC = """You are evaluating whether the response proposes a NOVELTY or SURPRISE-based filter.

**PASS if ANY of these concepts appear:**
- Words: "surprise", "surprising", "novel", "novelty", "unexpected", "anomaly"
- Words: "prediction error", "reconstruction error", "mismatch"
- Concept: Comparing what memory PREDICTS vs what actually arrives
- Concept: Storing things that are DIFFERENT from existing memory
- Concept: High error = high write, low error = low write
- Concept: "Interference" or "pattern saturation" as the problem (shows understanding)

**FAIL only if:**
- ONLY talks about "importance" without defining it as surprise
- ONLY uses attention-style "relevance" scoring
- Proposes filtering but with no concrete mechanism at all
- Says "select important tokens" without explaining what makes them important

Look for the WORDS "surprise", "novel", "unexpected", "error", "predict".
If any of these appear in context of filtering, PASS.

Respond with exactly:
STEP 2: PASS or FAIL
Reason: [Found surprise/novelty concept? Quote the key phrase.]"""

STEP_3_RUBRIC = """You are evaluating whether the update rule includes DECAY (forgetting).

**The key insight:** Memory needs to FORGET old information to prevent saturation.

**PASS if the equation includes DECAY:**
- Shows γ * M or (1-η) * M or decay * M or similar
- The decay factor should be < 1 (or learned)
- Can be written as: M_new = decay*M + update
- Or equivalently: M_new = M - forget_rate*M + update

**ALSO give PASS if:**
- Shows explicit "forgetting" or "decay" in the equation
- Uses exponential moving average style update
- Has a term that REDUCES old memory over time

**FAIL if:**
- Just M_new = M + update (purely additive, no decay)
- No explicit decay/forget term visible
- Says "decay" in text but equation shows M_new = M + ...

Look at the EQUATION specifically. Is there multiplication of M by a decay factor?

Respond with exactly:
STEP 3: PASS or FAIL
Reason: [Is there a decay term multiplying M_old? Quote it.]"""

STEP_4_RUBRIC = """You are evaluating whether fusion uses COMPLEMENTARY GATING (weights sum to 1).

**The key pattern:** out = g*attn + (1-g)*mem where g comes from sigmoid

**PASS requires ALL of these:**
1. Shows the explicit (1-g) or (1-gate) pattern - MUST be present
2. Uses sigmoid (not softmax, not hardcoded)
3. The gate g is learned (from a linear layer, not hardcoded 0.5)

**FAIL if ANY of these (be strict about complementary):**
- Simple addition: out = attn + mem (no gating)
- Concatenation only: out = Linear(concat(attn, mem)) (no g * (1-g))
- Hardcoded weights: 0.5*attn + 0.5*mem
- Two separate weights: α*attn + β*mem (unless α + β = 1 is enforced)
- Missing (1-g) pattern: g*attn + mem (only one term gated)
- Says "gating" but doesn't show the (1-g) in the formula
- Uses softmax over [attn, mem] (while valid, we specifically want sigmoid + (1-g))

The (1-g) is CRITICAL. Look for "1 - g", "(1-gate)", "1 - gate" explicitly in the output formula.

Respond with exactly:
STEP 4: PASS or FAIL
Reason: [Is (1-g) pattern explicit in the fusion equation?]"""

STEP_5_RUBRIC = """You are evaluating the implementation against specific requirements.

**4 required components (from the prompt):**
1. **Matrix M storage** - Memory stored as a matrix (d×d or d×k), NOT as list of vectors
2. **Outer product update** - Uses outer(v, k), v @ k.T, torch.bmm, or einsum for updates
3. **Saturation prevention** - Has decay (η*M), gating, or forgetting mechanism
4. **Token-by-token processing** - Explicit for loop over sequence (not batch processing)

**PASS requires ALL 4 components:**
- All 4 present = PASS
- Missing ANY = FAIL

**Strict checking:**
- Matrix storage: Look for M shape like (d, d) or (d_k, d_v), not (n, d)
- Outer product: Must show explicit transpose or outer operation
- Saturation: Must have decay term multiplying M or forget gate
- Sequential: Must have "for t in range" or similar loop

Most implementations miss 1-2 components. Be strict about the requirements stated in the prompt.

Respond with exactly:
STEP 5: PASS or FAIL
Components passed: X/4
Reason: [Which are present/missing?]"""

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


def _run_single_evaluation_quiet(model: str, run_id: int):
    """Run a single evaluation without printing (for parallel runs)."""
    client = anthropic.Anthropic()

    # Run all 5 steps in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, (step_name, prompt) in enumerate(PROMPTS):
            future = executor.submit(
                _run_single_step,
                client, model, i, step_name, prompt, RUBRICS[i]
            )
            futures.append(future)
        step_results = [f.result() for f in futures]

    step_results.sort(key=lambda x: x["step"])
    total_passed = sum(1 for s in step_results if s["passed"])
    overall_pass = total_passed >= 4

    return {
        "run_id": run_id,
        "model": model,
        "steps": step_results,
        "total_passed": total_passed,
        "overall_pass": overall_pass,
    }


def run_multiple_evaluations(model: str = "claude-haiku-4-5", num_runs: int = 5):
    """
    Run multiple evaluations IN PARALLEL and aggregate results.

    Args:
        model: The model to test
        num_runs: Number of evaluation runs (all run in parallel)

    Returns:
        dict with aggregated statistics
    """
    print(f"\n{'='*60}")
    print(f"PARALLEL EVALUATION: {num_runs} runs")
    print(f"Model: {model}")
    print(f"{'='*60}")
    print(f"\nRunning {num_runs} evaluations in parallel...")

    # Run all evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [
            executor.submit(_run_single_evaluation_quiet, model, i)
            for i in range(num_runs)
        ]
        all_results = [f.result() for f in futures]

    # Aggregate results
    step_pass_counts = [0, 0, 0, 0, 0]
    overall_pass_count = 0

    print(f"\n{'='*60}")
    print("INDIVIDUAL RUN RESULTS")
    print(f"{'='*60}")

    for result in all_results:
        steps_passed = [s["passed"] for s in result["steps"]]
        step_str = "".join(["✓" if p else "✗" for p in steps_passed])
        status = "PASS" if result["overall_pass"] else "FAIL"
        print(f"Run {result['run_id']+1}: {step_str} ({result['total_passed']}/5) - {status}")

        for i, passed in enumerate(steps_passed):
            if passed:
                step_pass_counts[i] += 1
        if result["overall_pass"]:
            overall_pass_count += 1

    # Print statistics
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")

    step_names = ["Memory Structure", "Filtering/Surprise", "Update Rule", "Fusion", "Implementation"]
    for i, (name, count) in enumerate(zip(step_names, step_pass_counts)):
        pct = count / num_runs * 100
        print(f"Step {i+1} ({name}): {count}/{num_runs} passed ({pct:.0f}%)")

    overall_pct = overall_pass_count / num_runs * 100
    print(f"\nOverall Pass Rate: {overall_pass_count}/{num_runs} ({overall_pct:.0f}%)")

    return {
        "model": model,
        "num_runs": num_runs,
        "step_pass_counts": step_pass_counts,
        "step_pass_rates": [c/num_runs for c in step_pass_counts],
        "overall_pass_count": overall_pass_count,
        "overall_pass_rate": overall_pass_count / num_runs,
        "all_results": all_results,
    }


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
    import sys

    # Check for --multi flag to run multiple parallel evaluations
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        results = run_multiple_evaluations("claude-haiku-4-5", num_runs=num_runs)
    else:
        # Single run with full output
        results = run_progressive_evaluation("claude-haiku-4-5")
