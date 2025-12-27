"""
Neural Memory Implementation Task

Tests whether an agent can implement a neural long-term memory module
from biological hints, getting all the technical details right.

Pass Rate: ~10% (Sonnet 4.5)
"""

import importlib.util
from pathlib import Path

# Load the combined task
task_path = Path(__file__).parent / "tasks" / "titan-paper-implementation" / "combined_task.py"

spec = importlib.util.spec_from_file_location("neural_memory_task", task_path)
neural_memory_task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_memory_task)

# Re-export the required interface
PROMPT = neural_memory_task.PROMPT
TOOLS = neural_memory_task.TOOLS
TOOL_HANDLERS = neural_memory_task.TOOL_HANDLERS
grading_func = neural_memory_task.grading_func
