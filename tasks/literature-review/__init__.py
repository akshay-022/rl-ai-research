"""Literature Review Task Package"""

from .task import PROMPT, TOOLS, TOOL_HANDLERS, TOPIC, grading_func
from .tools import web_search, get_paper_with_tldr, get_paper_references

__all__ = [
    "PROMPT",
    "TOOLS",
    "TOOL_HANDLERS",
    "TOPIC",
    "grading_func",
    "web_search",
    "get_paper_with_tldr",
    "get_paper_references",
]
