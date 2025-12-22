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
# Each prompt is SELF-CONTAINED and can be evaluated independently
# Later steps get brief Titans paper hints for necessary context

STEP_1_PROMPT = """I want to add a memory system to a Transformer that can handle very long sequences without the quadratic attention cost.

What kind of memory structure would you propose? Describe its mathematical form."""

STEP_2_PROMPT = """Consider a neural memory system where we store information in a weight matrix M that gets updated for every token.

If you update this memory for every token, won't it get overwhelmed with noise?

What's the problem here and how would you address it?"""

STEP_3_PROMPT = """Consider a neural memory system for Transformers where:
- Memory is stored as a matrix M (like fast weights)
- We need to control which tokens get written to memory

Give me the complete mathematical update rule for this memory.

Write it as a single equation: M_new = ..."""

STEP_4_PROMPT = """Consider a Transformer block that has BOTH:
1. Standard attention (for short-term context)
2. A neural memory module (for long-term storage)

How do you combine their outputs in a single block? The Titans paper uses a "Memory as a Gate" approach."""

STEP_5_PROMPT = """Implement a Titans-style neural memory module in PyTorch.

Context from the Titans paper:
- Memory is a matrix M that stores key-value associations via outer product: V ⊗ K^T
- A "surprise" gate (sigmoid) controls how much each token writes to memory
- Memory decays over time with a learnable parameter
- The update rule: M_new = decay * M_old + surprise * (V ⊗ K^T)

Give me a complete, runnable `NeuralMemory` module and a `MemoryBlock` that combines attention with memory using learned gating."""


PROMPTS = [
    ("Step 1: High-Dimensional Memory", STEP_1_PROMPT),
    ("Step 2: Surprise/Importance Filter", STEP_2_PROMPT),
    ("Step 3: Update Rule with Decay", STEP_3_PROMPT),
    ("Step 4: Attention-Memory Fusion", STEP_4_PROMPT),
    ("Step 5: Implementation", STEP_5_PROMPT),
]


# --- GRADING RUBRICS FOR EACH STEP ---

STEP_1_RUBRIC = """Evaluate if the response proposes a HIGH-DIMENSIONAL memory structure.

**PASS criteria (must meet at least one):**
- Proposes a "matrix" or "weight matrix" as memory state (not just a vector)
- Mentions "fast weights" or "dynamic linear layer"
- Suggests "outer product" to store key-value associations
- Proposes memory with dimensions like (d x d) or (memory_size x d_model)

**FAIL criteria:**
- Only proposes a larger hidden vector (e.g., "use a 4096-dim vector")
- Suggests just "more attention heads" or "more layers"
- Proposes standard LSTM/GRU without matrix-based memory
- No concrete mathematical structure proposed

Respond with exactly:
STEP 1: PASS or FAIL
Reason: [brief explanation]"""

STEP_2_RUBRIC = """Evaluate if the response proposes a FILTERING mechanism for memory updates.

**PASS criteria (must meet at least one):**
- Mentions "importance score" or "relevance gate"
- Proposes "surprise" or "novelty" detection
- Suggests using "prediction error" to filter updates
- Proposes a learned gate (sigmoid, softmax) to control write strength
- Mentions comparing input to existing memory to decide updates

**FAIL criteria:**
- Suggests writing every token equally to memory
- Only proposes periodic clearing/reset of memory
- Relies on external signals to decide what to remember
- No filtering mechanism proposed

Respond with exactly:
STEP 2: PASS or FAIL
Reason: [brief explanation]"""

STEP_3_RUBRIC = """Evaluate if the response proposes the correct UPDATE RULE with decay and surprise.

**PASS criteria (must meet ALL):**
1. Proposes approximating "surprise" without backprop (e.g., prediction error, reconstruction error, or learned gate)
2. Includes a DECAY term for forgetting (like η or (1-η) multiplied with old state)
3. The update rule has form: M_new = decay * M_old + surprise * new_data
   Or equivalent: M = (1-η)M + s·(V⊗K)

**PARTIAL PASS (3/5 points):**
- Has decay but no surprise mechanism
- Has surprise but no decay
- Formula structure is close but missing key component

**FAIL criteria:**
- Hard if/else rules instead of soft gating
- No decay/forgetting mechanism
- Cannot connect surprise to update magnitude
- No mathematical formula proposed

Respond with exactly:
STEP 3: PASS or FAIL
Reason: [brief explanation]"""

STEP_4_RUBRIC = """Evaluate if the response proposes correct FUSION/GATING of attention and memory.

**PASS criteria:**
- Proposes a LEARNED gate (not hardcoded)
- Uses sigmoid activation for soft gating
- Formula like: g = sigmoid(W[attn; mem]) then out = g*attn + (1-g)*mem
- Or equivalent weighted combination with learned parameters

**FAIL criteria:**
- Suggests simple addition: output = attention + memory
- Suggests concatenation without learned weighting
- Proposes alternating between sources (not blending)
- No gating mechanism proposed

Respond with exactly:
STEP 4: PASS or FAIL
Reason: [brief explanation]"""

STEP_5_RUBRIC = """Evaluate the FINAL IMPLEMENTATION code.

Grade each component (1 point each, need 5/6 to pass):

1. **NeuralMemory class** - Proper nn.Module with forward method
2. **Surprise gate** - Sigmoid-based learned gate for update strength
3. **Learnable decay** - nn.Parameter for memory fade
4. **Outer product** - V⊗K^T computed correctly (bmm or einsum)
5. **Recurrent loop** - Step-by-step updates (for t in range...)
6. **Fusion gate** - Learned gate combining attention and memory

Also check if they mention appropriate evaluation:
- "Needle in a Haystack" test
- BABILong benchmark
- Long-range retrieval tasks
- Passkey retrieval

Respond with exactly:
STEP 5: PASS or FAIL
Components passed: X/6
Evaluation mentioned: Yes/No
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
