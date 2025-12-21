"""FSDP Training Task Package"""

# Note: Use absolute imports when running from parent directory
# Use relative imports when imported as a package

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
