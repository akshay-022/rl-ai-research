#!/usr/bin/env python3
"""
Test Runner for FSDP Training Task

Run all tests and show summary.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py --quick   # Run only grading tests
    python run_tests.py --eval    # Run agent evaluation
"""

import argparse
import subprocess
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_unit_tests(quick=False):
    """Run unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    if quick:
        # Run only grading tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_grading.py::TestFSDPGrading", "-v"],
            capture_output=False
        )
    else:
        # Run all tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_grading.py", "-v"],
            capture_output=False
        )

    return result.returncode == 0


def run_grading_demo():
    """Demonstrate grading function on solution vs starter."""
    print("\n" + "=" * 60)
    print("Grading Function Demo")
    print("=" * 60)

    from task import grading_func

    # Test reference solution
    print("\n--- Testing Reference Solution ---")
    with open("train_fsdp_solution.py") as f:
        solution = f.read()
    solution_result = grading_func(solution)

    # Test starter code
    print("\n--- Testing Starter Code ---")
    with open("train_single_gpu.py") as f:
        starter = f.read()
    starter_result = grading_func(starter)

    print("\n" + "-" * 40)
    print("Summary:")
    print(f"  Reference Solution: {'PASS' if solution_result else 'FAIL'}")
    print(f"  Starter Code:       {'PASS' if starter_result else 'FAIL'}")

    return solution_result and not starter_result


def run_model_test():
    """Test the model can be instantiated and run."""
    print("\n" + "=" * 60)
    print("Model Sanity Check")
    print("=" * 60)

    try:
        import torch
        from model import SimpleTransformer, TransformerBlock, get_model_config

        config = get_model_config("tiny")
        print(f"Config: {config}")

        model = SimpleTransformer(**config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {num_params:,} parameters")

        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        print(f"Forward pass successful!")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")

        # Test backward pass
        outputs["loss"].backward()
        print(f"Backward pass successful!")

        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_evaluation(num_runs=1):
    """Run agent evaluation."""
    print("\n" + "=" * 60)
    print(f"Running Agent Evaluation ({num_runs} run(s))")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "evaluation.py", "-n", str(num_runs)],
        capture_output=False
    )

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="FSDP Task Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run only quick grading tests")
    parser.add_argument("--eval", action="store_true", help="Run agent evaluation")
    parser.add_argument("--eval-runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--no-unit", action="store_true", help="Skip unit tests")

    args = parser.parse_args()

    results = {}

    # Model test
    if not args.quick and not args.eval:
        results["model"] = run_model_test()

    # Grading demo
    if not args.eval:
        results["grading_demo"] = run_grading_demo()

    # Unit tests
    if not args.no_unit and not args.eval:
        try:
            import pytest
            results["unit_tests"] = run_unit_tests(quick=args.quick)
        except ImportError:
            print("\nWARNING: pytest not installed, running basic tests only")
            results["unit_tests"] = True  # Skip

    # Evaluation
    if args.eval:
        results["evaluation"] = run_evaluation(args.eval_runs)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
