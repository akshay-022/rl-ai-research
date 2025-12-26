"""
Unit Tests for FSDP Training Task Grading

Run with: python test_grading.py

NOTE: These tests use LLM-as-a-judge which requires API calls.
For quick unit tests without API calls, use test_grading_unit.py instead.
"""

import unittest
import os
import sys

# Ensure we can import from the task module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task import grading_func, _strip_markdown_wrapper, _check_code_executes


class TestCodeExecution(unittest.TestCase):
    """Test the code execution validation (no API calls needed)."""

    def test_valid_python_syntax(self):
        """Valid Python syntax should pass execution check."""
        valid_code = '''
import torch
import torch.distributed as dist

def train():
    print("hello")
'''
        ok, msg = _check_code_executes(valid_code)
        self.assertTrue(ok, f"Valid code should pass: {msg}")

    def test_invalid_python_syntax(self):
        """Invalid Python syntax should fail execution check."""
        invalid_code = """
def broken(
    print("missing parenthesis"
"""
        ok, msg = _check_code_executes(invalid_code)
        self.assertFalse(ok, "Invalid syntax should fail")

    def test_strip_markdown_wrapper(self):
        """Markdown wrapper should be stripped correctly."""
        code = "print('hello')"
        wrapped = f"```python\n{code}\n```"

        result = _strip_markdown_wrapper(wrapped)
        self.assertEqual(result, code)

    def test_strip_markdown_wrapper_no_lang(self):
        """Markdown wrapper without language should be stripped."""
        code = "print('hello')"
        wrapped = f"```\n{code}\n```"

        result = _strip_markdown_wrapper(wrapped)
        self.assertEqual(result, code)

    def test_no_markdown_wrapper(self):
        """Code without markdown wrapper should be unchanged."""
        code = "print('hello')"
        result = _strip_markdown_wrapper(code)
        self.assertEqual(result, code)


class TestGradingBasicChecks(unittest.TestCase):
    """Test basic grading checks that don't require LLM calls."""

    def test_empty_submission_fails(self):
        """Empty submission should fail."""
        result = grading_func("")
        self.assertFalse(result)

        result = grading_func(None)
        self.assertFalse(result)

    def test_short_code_fails(self):
        """Very short code should fail."""
        result = grading_func("print('hi')")
        self.assertFalse(result)


@unittest.skipIf(
    os.environ.get("SKIP_LLM_TESTS", "").lower() in ("1", "true", "yes"),
    "Skipping LLM-based tests (SKIP_LLM_TESTS is set)"
)
class TestFSDPGradingWithLLM(unittest.TestCase):
    """
    Test the full grading function with LLM-as-a-judge.

    These tests make API calls and are slower. Skip with SKIP_LLM_TESTS=1.
    """

    def test_reference_solution_passes(self):
        """Reference solution should pass all checks."""
        with open("train_fsdp_solution.py") as f:
            solution = f.read()

        result = grading_func(solution)
        self.assertTrue(result, "Reference solution should pass")

    def test_starter_code_fails(self):
        """Starter single-GPU code should fail."""
        with open("train_single_gpu.py") as f:
            starter = f.read()

        result = grading_func(starter)
        self.assertFalse(result, "Starter code should fail")

    def test_code_with_markdown_wrapper(self):
        """Code wrapped in markdown blocks should still be parsed."""
        with open("train_fsdp_solution.py") as f:
            solution = f.read()

        markdown_wrapped = f"```python\n{solution}\n```"
        result = grading_func(markdown_wrapped)
        self.assertTrue(result, "Markdown-wrapped code should pass")

    def test_ddp_instead_of_fsdp_fails(self):
        """Using DDP instead of FSDP should fail."""
        ddp_code = '''
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 10, (1,))

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = torch.nn.Linear(10, 10).cuda()
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters())

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            pass

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
'''
        result = grading_func(ddp_code)
        self.assertFalse(result, "DDP code should fail (not FSDP)")

    def test_partial_implementation_fails(self):
        """Partial implementation missing key components should fail."""
        partial_code = '''
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = torch.nn.Linear(10, 10)
    model = FSDP(model)

    # Missing: DistributedSampler, set_epoch, checkpoint handling with FSDP state dict, etc.

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
'''
        result = grading_func(partial_code)
        self.assertFalse(result, "Partial implementation should fail")


class TestModelDefinition(unittest.TestCase):
    """Test the model definition is correct."""

    def test_model_imports(self):
        """Model should be importable."""
        from model import SimpleTransformer, TransformerBlock, get_model_config
        self.assertTrue(callable(SimpleTransformer))
        self.assertTrue(callable(TransformerBlock))
        self.assertTrue(callable(get_model_config))

    def test_model_creation(self):
        """Model should be creatable with default config."""
        from model import SimpleTransformer, get_model_config

        config = get_model_config("tiny")
        model = SimpleTransformer(**config)

        # Check model has parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)

    def test_model_forward(self):
        """Model forward pass should work."""
        import torch
        from model import SimpleTransformer, get_model_config

        config = get_model_config("tiny")
        model = SimpleTransformer(**config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)

        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, config["vocab_size"]))

    def test_transformer_block_exists(self):
        """TransformerBlock should be a proper nn.Module."""
        import torch.nn as nn
        from model import TransformerBlock

        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512)
        self.assertIsInstance(block, nn.Module)


class TestStarterCode(unittest.TestCase):
    """Test the starter code is valid."""

    def test_starter_code_syntax(self):
        """Starter code should have valid Python syntax."""
        import ast

        with open("train_single_gpu.py") as f:
            code = f.read()

        # Should not raise
        ast.parse(code)

    def test_starter_code_has_train_function(self):
        """Starter code should have a train function."""
        with open("train_single_gpu.py") as f:
            code = f.read()

        self.assertIn("def train(", code)

    def test_starter_code_no_fsdp(self):
        """Starter code should NOT have FSDP imports or usage."""
        with open("train_single_gpu.py") as f:
            code = f.read()

        # Should not have FSDP imports (mentions in comments are OK)
        self.assertNotIn("from torch.distributed.fsdp", code)
        self.assertNotIn("FullyShardedDataParallel", code)
        self.assertNotIn("DistributedSampler", code)


class TestSolutionCode(unittest.TestCase):
    """Test the reference solution is valid."""

    def test_solution_syntax(self):
        """Solution should have valid Python syntax."""
        import ast

        with open("train_fsdp_solution.py") as f:
            code = f.read()

        ast.parse(code)

    def test_solution_has_fsdp(self):
        """Solution should have FSDP imports."""
        with open("train_fsdp_solution.py") as f:
            code = f.read()

        self.assertIn("FSDP", code)
        self.assertIn("DistributedSampler", code)
        self.assertIn("init_process_group", code)


class TestTaskPrompt(unittest.TestCase):
    """Test the task prompt is properly formed."""

    def test_prompt_contains_starter_code(self):
        """Prompt should contain the starter code."""
        from task import PROMPT

        self.assertIn("train_single_gpu.py", PROMPT)
        self.assertIn("def train(", PROMPT)

    def test_prompt_contains_requirements(self):
        """Prompt should mention key requirements."""
        from task import PROMPT

        self.assertIn("FSDP", PROMPT)
        self.assertIn("DistributedSampler", PROMPT)
        self.assertIn("TransformerBlock", PROMPT)
        self.assertIn("torchrun", PROMPT)

    def test_tools_defined(self):
        """Tools should be properly defined."""
        from task import TOOLS, TOOL_HANDLERS

        self.assertEqual(len(TOOLS), 1)
        self.assertEqual(TOOLS[0]["name"], "submit_answer")
        self.assertIn("submit_answer", TOOL_HANDLERS)


if __name__ == "__main__":
    # Change to the script's directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run tests with verbosity
    unittest.main(verbosity=2)
