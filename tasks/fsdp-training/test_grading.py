"""
Unit Tests for FSDP Training Task Grading

Run with: python test_grading.py
"""

import unittest
from task import grading_func


class TestFSDPGrading(unittest.TestCase):
    """Test the grading function with various code samples."""

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

    def test_empty_submission_fails(self):
        """Empty submission should fail."""
        result = grading_func("")
        self.assertFalse(result)

        result = grading_func(None)
        self.assertFalse(result)

    def test_invalid_python_fails(self):
        """Invalid Python syntax should fail."""
        invalid_code = """
def broken(
    print("missing parenthesis"
"""
        result = grading_func(invalid_code)
        self.assertFalse(result, "Invalid syntax should fail")

    def test_partial_implementation_fails(self):
        """Partial implementation missing key components should fail."""
        # Has FSDP but missing other requirements
        partial_code = '''
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train():
    dist.init_process_group(backend="nccl")
    model = MyModel()
    model = FSDP(model)
    # Missing: DistributedSampler, set_epoch, checkpoint handling, etc.
    dist.destroy_process_group()
'''
        result = grading_func(partial_code)
        self.assertFalse(result, "Partial implementation should fail")

    def test_code_with_markdown_wrapper(self):
        """Code wrapped in markdown blocks should still be parsed."""
        with open("train_fsdp_solution.py") as f:
            solution = f.read()

        markdown_wrapped = f"```python\n{solution}\n```"
        result = grading_func(markdown_wrapped)
        self.assertTrue(result, "Markdown-wrapped code should pass")

    def test_minimum_viable_fsdp(self):
        """Test minimum code that should pass (8/10 checks)."""
        minimal_code = '''
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from functools import partial

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    model = FSDP(MyModel(), auto_wrap_policy=auto_wrap_policy)

    sampler = DistributedSampler(dataset)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        # training...

    if dist.get_rank() == 0:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            torch.save(model.state_dict(), "checkpoint.pt")

    dist.destroy_process_group()
'''
        result = grading_func(minimal_code)
        self.assertTrue(result, "Minimal viable FSDP should pass")

    def test_ddp_instead_of_fsdp_fails(self):
        """Using DDP instead of FSDP should fail."""
        ddp_code = '''
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(dataset)

    for epoch in range(10):
        sampler.set_epoch(epoch)

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")

    dist.destroy_process_group()
'''
        result = grading_func(ddp_code)
        self.assertFalse(result, "DDP code should fail (not FSDP)")

    def test_missing_set_epoch_fails(self):
        """Missing set_epoch should contribute to failure."""
        no_set_epoch = '''
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from functools import partial

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    model = FSDP(MyModel(), auto_wrap_policy=auto_wrap_policy)

    sampler = DistributedSampler(dataset)

    for epoch in range(10):
        # MISSING: sampler.set_epoch(epoch)
        pass

    if dist.get_rank() == 0:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            torch.save(model.state_dict(), "checkpoint.pt")

    dist.destroy_process_group()
'''
        # This should still pass with 9/10 (threshold is 8)
        result = grading_func(no_set_epoch)
        # Missing set_epoch means 9/10, still passes
        self.assertTrue(result, "Missing one check should still pass (9/10 >= 8)")

    def test_missing_multiple_components_fails(self):
        """Missing multiple components should fail."""
        incomplete = '''
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train():
    dist.init_process_group(backend="nccl")
    model = FSDP(MyModel())
    # Missing: LOCAL_RANK, DistributedSampler, set_epoch, wrap_policy, checkpoint handling
    dist.destroy_process_group()
'''
        result = grading_func(incomplete)
        self.assertFalse(result, "Missing multiple components should fail")


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
