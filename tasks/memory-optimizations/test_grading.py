"""
Test the grading function with the reference solution.
"""

from pathlib import Path
from task import grading_func

# Read the solution file
solution_path = Path(__file__).parent / "train_optimized_solution.py"
with open(solution_path) as f:
    solution_code = f.read()

print("Testing grading function with reference solution...")
print("=" * 60)

result = grading_func(solution_code)

print("\n" + "=" * 60)
if result:
    print("✓ Reference solution PASSED grading")
else:
    print("✗ Reference solution FAILED grading - need to fix!")


# Also test that baseline fails
print("\n\nTesting that baseline code FAILS grading...")
print("=" * 60)

baseline_path = Path(__file__).parent / "train_baseline.py"
with open(baseline_path) as f:
    baseline_code = f.read()

baseline_result = grading_func(baseline_code)

print("\n" + "=" * 60)
if not baseline_result:
    print("✓ Baseline correctly FAILED grading (as expected)")
else:
    print("✗ Baseline should have failed but passed - grading too lenient!")
