hello-py
===

Setup instructions:

1. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```
   pip3 install .
   ```

3. Run the agent:
   ```
   python3 main.py
   ```

4. Run the tests:
   ```
   pytest
   ```

5. All required changes belong in `task.py`

## See transcript outputs

By default, `verbose=True` in `main.py`. Set `verbose=False` to turn off transcript outputs.

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

## Literature Review Task

An agent that conducts automated literature reviews by searching papers, extracting key sections, and synthesizing findings.

### How it works

1. **Agent Tools**: The agent has access to:
   - `web_search(query)` - Search arXiv/Semantic Scholar for papers
   - `get_paper_with_tldr(paper_id)` - Get paper details + AI-generated summary
   - `get_paper_introduction(paper_id)` - Extract intro section (contains authors' own lit review)
   - `get_paper_results(paper_id)` - Extract results/experiments section
   - `get_paper_references(paper_id)` - Get papers cited by a paper

2. **Agent Loop**: Haiku iteratively searches, reads papers, and builds understanding until it submits a complete review

3. **LLM-as-Judge Evaluation**: Sonnet grades the review against a reference using a strict 10-point rubric:
   - **Section 1 (5 pts)**: Coverage of all major approaches (must include specific papers + quantitative results)
   - **Section 2 (3 pts)**: Seminal works cited with results
   - **Section 3 (2 pts)**: Quality of synthesis (comparisons, trade-offs, trends)
   - Pass threshold: 7/10

### Topics Covered

The current literature reviews cover:
- **LLM Memory**: Extended context, RAG, summarization, MemGPT, model editing (ROME/MEMIT)
- **RL & Alignment**: RLHF, continual RL, world models, Decision Transformer
- **Robotics**: Foundation models (RT-1/RT-2), sim-to-real, imitation learning

### Key Design Decisions

- **Extract introductions, not just abstracts**: Paper intros contain the authors' own literature context
- **Require quantitative results**: "RAG helps" is not enough - need specific numbers
- **Reference-based grading**: Compare against ground truth literature review
- **Strict rubric**: Each criterion has clear PASS/FAIL conditions

## Idea Proposal Task

Generates novel research ideas from literature reviews using a Haiku agent loop, then evaluates them with Sonnet extended thinking.

### Usage

```bash
cd tasks/idea-proposal

# Full mode: reads all 3 literatures (RL, Memory, Robotics)
python test_task.py

# Single topic modes:
python test_task.py --memory     # LLM long-term memory
python test_task.py --rl         # Reinforcement learning & alignment
python test_task.py --robotics   # Data-efficient robotics
```

### How it works

1. **Haiku Agent Loop**: Reads literature reviews, optionally searches for papers, generates 3-5 research ideas
2. **Sonnet Extended Thinking Judge**: Evaluates each idea on:
   - Business Viability (30%)
   - Potential Upside (25%)
   - Likelihood of Success (30%)
   - Novelty (15%)
3. **Scoring**: Each idea gets a weighted score, plus an average across all ideas

### Output

- Individual idea scores (1-10)
- Average score across all ideas
- Verdict: EXCEPTIONAL / GOOD / MEDIOCRE / POOR

## Titans Paper Implementation Task

Tests whether an agent can implement a research paper architecture from a vague description, getting all the technical details right.

### The Challenge

Given a simple Transformer and the instruction to implement the **Titans architecture** ("Learning to Memorize at Test Time" - Google Research), the agent must produce working code with:

1. **NeuralMemory Class** - A proper nn.Module with memory logic
2. **Surprise Gate with Sigmoid** - Learned metric controlling memory writes
3. **Learnable Decay Parameter** - nn.Parameter for memory fade rate
4. **Outer Product for Association** - V ⊗ K^T for key-value binding
5. **Recurrent Memory Update** - Per-timestep loop updating memory state
6. **Memory-Attention Gating** - Learned fusion of attention and memory outputs

### Grading

- **LLM-as-Judge** (Sonnet) evaluates code against 6 strict criteria
- **Pass threshold**: 6/6 (all criteria must pass)
- Reference solution provided in `titans_solution.py`

### Run Tests

```bash
cd tasks/titan-paper-implementation
python test_grading.py
```

## FSDP Training Task

Tests whether an agent can convert a single-GPU training script to distributed training with FSDP.

### The Prompt

The agent receives a single-GPU training script and is asked:

> Convert the training script to use PyTorch FSDP. Your solution must:
> 1. Initialize distributed training with NCCL backend
> 2. Wrap the model with FSDP using a custom wrapping policy
> 3. Use DistributedSampler to ensure each GPU gets different data
> 4. Handle checkpointing correctly (only save on rank 0)
> 5. Clean up the process group at the end
>
> **IMPORTANT**: Use `lambda_auto_wrap_policy` with a custom lambda function that wraps modules based on:
> - Module type: wrap all `TransformerBlock` layers
> - Parameter count: wrap any module with more than 1,000,000 parameters
> - Do NOT use `transformer_auto_wrap_policy` - implement the logic yourself

### Results

**100% success rate** - Haiku consistently produces correct FSDP implementations.

### Grading

- **Regex-based checks** for 11 required patterns
- **Pass threshold**: 9/11 checks
- Reference solution in `train_fsdp_solution.py`

### Run Tests

```bash
cd tasks/fsdp-training
python test_grading.py
```

## Memory Optimizations Task

Tests whether an agent can apply multiple GPU memory optimization techniques to reduce training memory footprint.

### The Challenge

Given a baseline training script, apply memory optimizations. Must include:

**Baseline (required)**:
- Mixed Precision Training (AMP)
- Gradient Checkpointing
- Scaled Dot Product Attention (FlashAttention)

**Additional (need 2+ to beat baseline)**:
- Gradient Accumulation
- Memory-efficient Optimizers (8-bit Adam, Adafactor)
- Explicit Memory Management (empty_cache, del tensors)
- CPU Offloading / Pin Memory
- torch.compile
- In-place Operations

### Grading

- **Regex-based checks** for optimization patterns
- Must have all 3 baseline + at least 2 additional
- Reference solution in `train_optimized_solution.py`

### Run Tests

```bash
cd tasks/memory-optimizations
python test_grading.py
```

---

## Task Summary

| Task | Purpose | Grading Method | Pass Threshold |
|------|---------|----------------|----------------|
| Literature Review | Research synthesis | LLM-as-Judge (Sonnet) | 7/10 |
| Idea Proposal | Novel idea generation | Extended Thinking Judge | GOOD or EXCEPTIONAL |
| Titans Implementation | Paper → Code | LLM-as-Judge (Sonnet) | 6/6 |
| FSDP Training | Distributed training | Regex checks | 9/11 |
| Memory Optimizations | GPU memory reduction | Regex checks | 3/3 baseline + 2 additional |

All reference solutions and test scripts are included in each task directory.