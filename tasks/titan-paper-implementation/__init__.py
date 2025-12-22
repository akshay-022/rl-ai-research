"""Titans Architecture Implementation Task Package"""

try:
    from .task import PROMPT, TOOLS, TOOL_HANDLERS, grading_func
except ImportError:
    from task import PROMPT, TOOLS, TOOL_HANDLERS, grading_func

__all__ = [
    "PROMPT",
    "TOOLS",
    "TOOL_HANDLERS",
    "grading_func",
]
