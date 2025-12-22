"""
Progressive Titans Evaluation Suite

Tests an agent's ability to "discover" the Titans architecture step-by-step,
rather than just implementing from a full specification.

Each step builds on the previous, testing whether the agent can independently
arrive at the key insights that make Titans work.
"""

import os
from typing import Any
from dotenv import load_dotenv
import anthropic

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


# --- THE 5-STEP PROGRESSIVE PROMPTS ---

STEP_1_PROMPT = """I want to modify the standard Transformer to handle effectively infinite context without increasing the quadratic cost of attention.

Standard RNNs solve this with a 'hidden state,' but that vector is too small and low-dimensional to hold complex historical details. I want you to propose a **new memory module** that sits alongside the attention mechanism.

**Constraint:** This memory should be 'high-dimensional'—mathematically richer than a simple vector—and should act like a learning system that updates itself as it reads text.

Describe the mathematical structure of this memory state and how it 'stores' a new token."""

STEP_2_PROMPT = """That sounds plausible. However, we have a problem. If this memory module updates its state for every single token (like 'the', 'is', 'at'), the high-dimensional state will become saturated with noise and overwrite important information almost immediately.

Without using an external clear-memory trigger, **how does the module intrinsically decide** which information is worth 'encoding' into the long-term memory and which should be ignored?

Propose a mechanism that allows the model to self-regulate this update process."""

STEP_3_PROMPT = """Let's refine that filtering mechanism using an analogy to the human brain. We don't remember everything; we primarily remember things that are **unexpected** or **surprising** (i.e., where our internal prediction was wrong).

1. How can we mathematically approximate 'surprise' in a forward pass without running a full backward propagation?
2. The brain also slowly forgets old data to make room for new data.

Combine these two concepts into a single mathematical **Update Rule** for your memory matrix M. It should include terms for the **Old State**, the **Decay**, the **Surprise**, and the **New Data**."""

STEP_4_PROMPT = """Now we have two parallel branches in our block:
1. **Short-term:** The Sliding Window Attention (precise, local context).
2. **Long-term:** Your Neural Memory Matrix (compressed, infinite context).

Sometimes the answer is in the immediate past (Attention wins). Sometimes it's deep in the history (Memory wins).

If we just add their outputs together, the signal gets muddy. Design a **Fusion Layer** that dynamically resolves conflicts between these two branches. It should allow the model to 'choose' which memory source to trust for each specific token."""

STEP_5_PROMPT = """This architecture looks solid. Now, I need you to make it real.

1. **Code:** Write the PyTorch implementation for this `TitansBlock` and the `NeuralMemory` module. Ensure the 'Surprise' gate and 'Momentum/Decay' update rule are explicitly implemented.

2. **Evaluation:** Standard benchmarks (like MMLU) won't test this 'infinite memory' capability well. What specific **synthetic tasks** or **metrics** should we use to prove that this specific architecture is better than a standard Transformer?

Submit complete, runnable PyTorch code."""


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


def run_progressive_evaluation(model: str = "claude-haiku-4-5-20250514"):
    """
    Run the 5-step progressive evaluation on a model.

    Args:
        model: The model to test (e.g., "claude-haiku-4-5-20250514")

    Returns:
        dict with results for each step
    """
    client = anthropic.Anthropic()

    results = {
        "model": model,
        "steps": [],
        "conversation": [],
        "total_passed": 0,
    }

    conversation_history = []

    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TITANS EVALUATION")
    print(f"Model: {model}")
    print(f"{'='*60}")

    for i, (step_name, prompt) in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"{step_name}")
        print(f"{'='*60}")
        print(f"\n--- Prompt ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

        # Add user message to conversation
        conversation_history.append({
            "role": "user",
            "content": prompt
        })

        # Get response from model being tested
        print(f"\n--- {model} Response ---")
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=conversation_history
        )

        agent_response = response.content[0].text
        print(agent_response[:1500] + "..." if len(agent_response) > 1500 else agent_response)

        # Add assistant response to conversation
        conversation_history.append({
            "role": "assistant",
            "content": agent_response
        })

        # Grade this step using Sonnet
        print(f"\n--- Grading Step {i+1} ---")

        grade_prompt = f"""Grade the following response against the rubric.

RESPONSE TO GRADE:
{agent_response}

RUBRIC:
{RUBRICS[i]}"""

        grade_response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": grade_prompt
            }]
        )

        grade_text = grade_response.content[0].text
        print(grade_text)

        # Parse result
        passed = "pass" in grade_text.lower().split("\n")[0]

        step_result = {
            "step": i + 1,
            "name": step_name,
            "passed": passed,
            "response": agent_response,
            "grade": grade_text,
        }
        results["steps"].append(step_result)

        if passed:
            results["total_passed"] += 1

        results["conversation"] = conversation_history.copy()

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    for step in results["steps"]:
        status = "✓ PASS" if step["passed"] else "✗ FAIL"
        print(f"Step {step['step']}: {step['name']} - {status}")

    print(f"\nTotal: {results['total_passed']}/5 steps passed")

    # Overall pass requires at least 4/5 steps
    overall_pass = results["total_passed"] >= 4
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'} (need 4/5)")

    results["overall_pass"] = overall_pass

    return results


def run_comparison(models: list[str] = None):
    """
    Run the evaluation on multiple models and compare results.
    """
    if models is None:
        models = [
            "claude-haiku-4-5-20250514",
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
