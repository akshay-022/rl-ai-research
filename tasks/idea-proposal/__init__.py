"""
Idea Proposal Task

This task reads literature review datasets and generates novel research ideas,
then evaluates them using a deep-thinking LLM-as-judge approach.
"""

from .task import PROMPT, TOOLS, TOOL_HANDLERS, grading_func

__all__ = ["PROMPT", "TOOLS", "TOOL_HANDLERS", "grading_func"]
