"""
Tools for idea proposal task.
Provides access to literature review datasets for each research area.
"""

import os

# Get the dataset directory path
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'literature-review', 'dataset')


def _load_dataset_file(filename: str) -> str:
    """Load a dataset file from the literature-review dataset directory."""
    filepath = os.path.join(DATASET_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return f"Error: Could not find {filename}"


def get_rl_literature() -> dict:
    """
    Get the literature review on Reinforcement Learning and Model Alignment.

    This covers:
    - RLHF and instruction alignment
    - Continual and lifelong RL
    - Model-based RL and world models
    - Sequence modeling for RL (Decision Transformer)
    - Integrating LLMs with RL
    """
    content = _load_dataset_file('rl.md')
    return {
        "topic": "Reinforcement Learning and Model Alignment",
        "content": content,
        "error": None if not content.startswith("Error:") else content
    }


def get_memory_literature() -> dict:
    """
    Get the literature review on Long-Term Memory for LLMs.

    This covers:
    - Extended context windows (Gemini 1M, RMT)
    - Retrieval-Augmented Generation (RAG, RETRO)
    - Summarization and context compression
    - Specialized memory architectures (MemGPT, LongMem)
    - Parametric memory and model editing (ROME, MEMIT)
    """
    content = _load_dataset_file('memory.md')
    return {
        "topic": "Long-Term Memory for Large Language Models",
        "content": content,
        "error": None if not content.startswith("Error:") else content
    }


def get_robotics_literature() -> dict:
    """
    Get the literature review on Data-Efficient and Continual Robotics Learning.

    This covers:
    - Foundation models for robotics (RT-1, RT-2, PaLM-E)
    - Simulation and sim-to-real transfer
    - Self-supervised learning for robots
    - Imitation learning and human-in-the-loop
    - Continual/lifelong learning in robotics
    """
    content = _load_dataset_file('robotics.md')
    return {
        "topic": "Data-Efficient and Continual Robotics Learning",
        "content": content,
        "error": None if not content.startswith("Error:") else content
    }


def get_all_literature() -> dict:
    """
    Get all three literature reviews combined for cross-domain ideation.

    Returns literature on:
    1. Reinforcement Learning and Model Alignment
    2. Long-Term Memory for LLMs
    3. Data-Efficient and Continual Robotics Learning
    """
    rl = get_rl_literature()
    memory = get_memory_literature()
    robotics = get_robotics_literature()

    return {
        "topics": [
            {"name": rl["topic"], "content": rl["content"]},
            {"name": memory["topic"], "content": memory["content"]},
            {"name": robotics["topic"], "content": robotics["content"]},
        ],
        "error": None
    }


# Tool definitions for Anthropic API
GET_RL_LITERATURE_TOOL = {
    "name": "get_rl_literature",
    "description": "Get the comprehensive literature review on Reinforcement Learning and Model Alignment. Covers RLHF, continual RL, world models, Decision Transformer, and LLM+RL integration.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GET_MEMORY_LITERATURE_TOOL = {
    "name": "get_memory_literature",
    "description": "Get the comprehensive literature review on Long-Term Memory for LLMs. Covers extended context windows, RAG, summarization, memory architectures (MemGPT), and model editing.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GET_ROBOTICS_LITERATURE_TOOL = {
    "name": "get_robotics_literature",
    "description": "Get the comprehensive literature review on Data-Efficient and Continual Robotics Learning. Covers foundation models (RT-1, RT-2), sim-to-real, imitation learning, and lifelong learning.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GET_ALL_LITERATURE_TOOL = {
    "name": "get_all_literature",
    "description": "Get ALL three literature reviews at once for cross-domain ideation. Use this to find connections between RL, memory, and robotics research.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

TOOLS = [GET_RL_LITERATURE_TOOL, GET_MEMORY_LITERATURE_TOOL, GET_ROBOTICS_LITERATURE_TOOL, GET_ALL_LITERATURE_TOOL]

HANDLERS = {
    "get_rl_literature": lambda: get_rl_literature(),
    "get_memory_literature": lambda: get_memory_literature(),
    "get_robotics_literature": lambda: get_robotics_literature(),
    "get_all_literature": lambda: get_all_literature(),
}
