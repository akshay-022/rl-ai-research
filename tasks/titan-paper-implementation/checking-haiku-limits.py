"""
Test Haiku's ability to implement Titans architecture.

This script:
1. Sends the starter code and Titans spec to Haiku
2. Sends Haiku's output to Sonnet for evaluation
3. Also runs through our grading function
"""

import os
from dotenv import load_dotenv

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import anthropic
from task import PROMPT, grading_func

HAIKU_PROMPT = PROMPT + "\n\nReturn ONLY the complete Python code, no explanations."

SONNET_EVAL_PROMPT = """You are evaluating a Titans architecture implementation. Check if this code correctly implements the Neural Memory mechanism from the Titans paper.

Grade each of these 6 requirements (1 point each):
1. NeuralMemory class is defined
2. Surprise gate is implemented using sigmoid (torch.sigmoid or similar on a gate/surprise variable)
3. Learnable decay parameter exists (nn.Parameter with tensor)
4. Outer product uses transpose (like k_t.transpose(1, 2) for V @ K.T)
5. Recurrent memory state update (state = decay * state + ...)
6. Block fusion/gating combines attention and memory outputs (torch.cat or weighted combination)

Code to evaluate:
```python
{code}
```

For each requirement, say PASS or FAIL with a brief reason.
Then give a final score out of 6.
Finally, state if this would be considered a PASSING implementation (5+ points)."""


def main():
    client = anthropic.Anthropic()

    print("=" * 60)
    print("STEP 1: Asking Haiku to implement Titans")
    print("=" * 60)

    haiku_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": HAIKU_PROMPT
        }]
    )

    haiku_code = haiku_response.content[0].text

    # Extract code if wrapped in markdown
    if "```python" in haiku_code:
        haiku_code = haiku_code.split("```python")[1].split("```")[0]
    elif "```" in haiku_code:
        haiku_code = haiku_code.split("```")[1].split("```")[0]

    print("\n--- Haiku's Titans Code (first 2000 chars) ---")
    print(haiku_code[:2000] + "..." if len(haiku_code) > 2000 else haiku_code)

    print("\n" + "=" * 60)
    print("STEP 2: Asking Sonnet to evaluate")
    print("=" * 60)

    sonnet_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": SONNET_EVAL_PROMPT.format(code=haiku_code)
        }]
    )

    evaluation = sonnet_response.content[0].text

    print("\n--- Sonnet's Evaluation ---")
    print(evaluation)

    print("\n" + "=" * 60)
    print("STEP 3: Running through our grading function")
    print("=" * 60)

    result = grading_func(haiku_code)
    print(f"\nOur grading result: {'PASS' if result else 'FAIL'}")

    print("\n" + "=" * 60)
    print("FULL HAIKU OUTPUT (for reference)")
    print("=" * 60)
    print(haiku_code)


if __name__ == "__main__":
    main()
