"""
Unit tests for Titans task grading.
"""

import unittest
import sys
import os

# Add task directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task import grading_func, PROMPT, TOOLS


class TestTitansGrading(unittest.TestCase):
    """Test the grading function."""

    def test_reference_solution_passes(self):
        """The reference solution should pass grading."""
        with open("titans_solution.py") as f:
            code = f.read()
        self.assertTrue(grading_func(code))

    def test_empty_submission_fails(self):
        """Empty submission should fail."""
        self.assertFalse(grading_func(""))
        self.assertFalse(grading_func(None))

    def test_invalid_python_fails(self):
        """Invalid Python syntax should fail."""
        self.assertFalse(grading_func("def broken("))
        self.assertFalse(grading_func("class Foo("))

    def test_starter_code_fails(self):
        """Starter code without Titans modifications should fail."""
        starter_code = '''
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        b, s = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(torch.arange(s, device=input_ids.device))
        x = self.dropout(x)
        mask = torch.triu(torch.ones(s, s, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.output(self.norm(x))
        return {"logits": logits}
'''
        self.assertFalse(grading_func(starter_code))

    def test_partial_implementation_fails(self):
        """Partial implementation with just NeuralMemory class should fail."""
        partial_code = '''
import torch
import torch.nn as nn

class NeuralMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x
'''
        self.assertFalse(grading_func(partial_code))

    def test_code_with_markdown_wrapper(self):
        """Code wrapped in markdown should still be processed."""
        with open("titans_solution.py") as f:
            code = f.read()
        markdown_wrapped = f"```python\n{code}\n```"
        self.assertTrue(grading_func(markdown_wrapped))


class TestTaskPrompt(unittest.TestCase):
    """Test the task prompt is properly configured."""

    def test_prompt_contains_starter_code(self):
        """Prompt should contain the starter code."""
        self.assertIn("class TransformerBlock", PROMPT)
        self.assertIn("class SimpleTransformer", PROMPT)

    def test_prompt_contains_requirements(self):
        """Prompt should contain key requirements."""
        self.assertIn("NeuralMemory", PROMPT)
        self.assertIn("TitansBlock", PROMPT)
        self.assertIn("Surprise", PROMPT)
        self.assertIn("decay", PROMPT.lower())

    def test_tools_defined(self):
        """Tools should be properly defined."""
        self.assertEqual(len(TOOLS), 1)
        self.assertEqual(TOOLS[0]["name"], "submit_answer")


class TestSolutionCode(unittest.TestCase):
    """Test the reference solution is valid."""

    def test_solution_syntax(self):
        """Solution should have valid Python syntax."""
        with open("titans_solution.py") as f:
            code = f.read()
        import ast
        ast.parse(code)

    def test_solution_has_neural_memory(self):
        """Solution should define NeuralMemory class."""
        with open("titans_solution.py") as f:
            code = f.read()
        self.assertIn("class NeuralMemory", code)

    def test_solution_has_titans_block(self):
        """Solution should define TitansBlock class."""
        with open("titans_solution.py") as f:
            code = f.read()
        self.assertIn("class TitansBlock", code)

    def test_solution_runs(self):
        """Solution should run without errors."""
        import torch
        # Import the solution module
        import titans_solution
        model = titans_solution.SimpleTransformer(
            vocab_size=100, d_model=64, n_heads=2, n_layers=1, d_memory=32
        )
        input_ids = torch.randint(0, 100, (1, 16))
        output = model(input_ids)
        self.assertIn("logits", output)
        self.assertEqual(output["logits"].shape, (1, 16, 100))


if __name__ == "__main__":
    unittest.main()
