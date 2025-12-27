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

# Final Documentation

I was asked to create RL environments for an automated AI researcher. I found this assignment quite enjoyable, this is relevant to making AI disover new science to push humanity forward and I find that very exciting.

Instead of just trying to implement a single coding task in pytorch that the model fails at, I wanted to run many experiments along multiple dimensions of research competency to see how good AI is at it currently.

Let me first talk about my main submission and then all the other experiments I ran.

---

## Main Submission: Neural Memory Implementation Task

**Pass Rate: 10% (Sonnet 4.5)**

This is the biggest unhobbling needed to make AI superhuman, all current RL is cope to not have algorithms to do this well.

Moreover, I believe these continual learning models must have memory and it MUST be in the model weights, and not simply in stored context. The model weights are much higher dimensional, that is also the way our brain works (memory is stored in synapse weights).

I really like the paper Titans: Learning to Memorize at Test Time (by Google) - https://arxiv.org/abs/2501.00663, so I use that to give hints and get to design experiments. Verified based on the implementation in the paper.

### The Challenge

Given a simple transformer and hints about how biological memory works, the agent must implement a Neural Long-Term Memory module with 6 specific components:

1. **NeuralMemory Class** - A proper nn.Module with memory logic
2. **Surprise Gate with Sigmoid** - Learned metric controlling memory writes
3. **Learnable Decay Parameter** - nn.Parameter for memory fade rate
4. **Outer Product for Association** - V ⊗ K^T for key-value binding
5. **Recurrent Memory Update** - Per-timestep loop updating memory state
6. **Memory-Attention Gating** - Learned fusion: `g * attn + (1-g) * mem`

### The Prompt

```
Standard attention has quadratic cost and a fixed context window. Your goal is to add an auxiliary memory system that can store and retrieve information beyond the attention window.

Think about how biological memory works:
- The brain doesn't remember everything equally - it prioritizes surprising or novel information
- Old memories fade over time unless reinforced
- Memory involves associating concepts together (binding keys to values)
- We blend immediate context with long-term knowledge

When does a human update their "brain weights"? Your memory system should capture this principle.

Requirements:
1. Create a `NeuralMemory` module that maintains a memory matrix M (shape: d_mem x d_mem)
2. Process tokens recurrently (one at a time), updating M at each step
3. Use outer products to store associations: M += V ⊗ K^T binds values to keys
4. Preserve the input/output signature (`input_ids` -> `logits`)

Important: Make parameters learnable (nn.Parameter) rather than hardcoding values. Use sigmoid activations for gating. The model should be able to learn the right behavior through training.
```

### Why This Task Matters

The prompt gives biological hints (surprising info, fading memories, associations) and the outer product formula, but does NOT explicitly tell the model about surprise gates or fusion formulas. The model must derive these from first principles.

This was a task to give a model a research direction and ask it to implement the experiment while getting all the pesky details right. 

### Results

Models consistently get 4 out of 6 criteria right: the NeuralMemory class (explicitly required), learnable decay (prompted with "make parameters learnable"), outer product (formula given: `M += V ⊗ K^T`), and recurrent update ("process tokens one at a time").

**Where models fail:**

The **Memory-Attention Gating** (~80% failure rate) is the hardest criterion. Models use residual patterns like `output = x + gate * mem` or concatenation `output = Linear(concat(attn, mem))` instead of the complementary interpolation `g * attn + (1-g) * mem`. The `(1-g)` pattern is the key insight models miss.

The **Surprise Gate** (~50% failure rate) is the other differentiator. They implement generic learned gates like `write_gate = sigmoid(Linear(x))` instead of computing prediction-error based surprise by comparing input to what memory predicts (e.g., `surprise = sigmoid(net(x - retrieved))`). The prompt hints at "surprising or novel information" but doesn't say HOW to compute surprise.

**When it passed (1/10 runs):** The successful submission implemented a `novelty_net` comparing input with retrieved memory for surprise, and proper gating with `g * mem + (1-g) * x` formula.

### Ablations

I ran several ablations to calibrate the task difficulty:

| Prompt Configuration | Pass Rate | Notes |
|---------------------|-----------|-------|
| Explicit hints for all 6 criteria | 60-100% | Too easy - spoonfeeding the answer |
| Only biological hints (no outer product) | 0% | Too hard - models propose K,V caches instead of associative matrix |
| Biological hints + outer product formula | 0% | Still too hard without learnable params hint |
| + "Make parameters learnable" + "Use sigmoid for gating" | **10%** | Target range achieved |

The key insight: giving the outer product formula gets models to the right memory structure, but they still need to derive the surprise mechanism and fusion pattern themselves. That's what makes it a 10% task instead of 0% or 60%.

### Sister Experiment: Progressive 5-Step Evaluation

I also ran a decomposed version where instead of asking for the full implementation, I tested each concept independently in parallel:

| Step | Concept | Without Pseudocode | With Pseudocode |
|------|---------|-------------------|-----------------|
| 1 | Memory Structure (associative matrix) | 0% | 0% |
| 2 | Filtering/Surprise mechanism | 20% | 90% |
| 3 | Update Rule with Decay | 10% | 30% |
| 4 | Attention-Memory Fusion | 90% | 70% |
| 5 | Full Implementation | 70% | 90% |

**Key Insight**: Asking for pseudocode dramatically improves Steps 2, 3, and 5 because it forces the model to commit to concrete mechanisms rather than vague descriptions. Step 1 remains at 0% because the fundamental architectural choice (associative matrix vs. K,V cache) isn't affected by asking for pseudocode - it's a conceptual gap.

---

## Other Experiments

I wanted to test the limits of AI in how well it can do literature survey, how good the research directions it proposes are, and how well it can do infra optimizations to run experiments.

### Task 2: Literature Review

**Pass Rate: 100%**

Tests how comprehensively an LLM can survey a research field in 3 areas - memory, RL and robotics.

**The Prompt:**
```
Create a comprehensive literature review covering ALL major approaches:
- Extended context windows (Gemini 1M, Longformer, RMT)
- Retrieval-Augmented Generation (RAG, RETRO)
- Summarization and context compression
- Specialized memory architectures (MemGPT, LongMem)
- Parametric memory (ROME/MEMIT, continual learning)

For EACH approach, include specific papers with quantitative results.
```

**Tools provided:**
- `web_search(query)` - Search arXiv/Semantic Scholar for papers
- `get_paper_with_tldr(paper_id)` - Get paper details + AI-generated summary
- `get_paper_introduction(paper_id)` - Extract intro section (contains authors' lit review)
- `get_paper_results(paper_id)` - Extract quantitative results from PDF
- `get_paper_references(paper_id)` - Get papers cited by a paper

**Evaluation:** LLM-as-Judge (Sonnet) grades against a 10-point rubric requiring specific papers + quantitative results for each approach (Passes if 7/10 or above). I created deep research reports with detailed guidelines for each field (Memory, RL, Robotics) as ground truth, and the rubric compares the agent's output against these reference documents. Mainly graded based on if it captures the right high level insights and results from papers, and if it references all the seminal papers. 

**Result:** Agent produced 28K-character review with 20+ papers. Successfully found LongLoRA (4k→100k context), Infini-attention (96-100% at 1M tokens), MemGPT (92.5% vs 32.1% baseline).

---

### Task 3: Idea Proposal

**Pass Rate: 5.6/10**

Tests research taste - can the model propose ideas that are both novel AND likely to succeed.

**The Prompt:**
```
You have access to literature reviews in RL, Memory, and Robotics.
Propose 5-7 novel research ideas that:
- Address REAL gaps in the literature
- Have HIGH IMPACT if successful
- Have BUSINESS VALUE (practical applications)
- Are FEASIBLE within 1-2 years

For each idea, provide: Problem, Approach, Why Now, Impact/Business/Feasibility scores, Key Risks.
Prioritize by: (Impact * 0.4) + (Business Value * 0.4) + (Feasibility * 0.2)
```

**Tools provided:**
- `get_all_literature()` - Read all 3 literature reviews
- `get_literature(topic)` - Read specific topic (RL, Memory, Robotics)
- `web_search(query)` - Search for additional papers

**Evaluation:** Sonnet with extended thinking (~150s reasoning, 10k thinking tokens) judges each idea on:
- Business Viability (30%)
- Potential Upside (25%)
- Likelihood of Success (30%)
- Novelty (15%)

Pass requires GOOD or EXCEPTIONAL verdict.

**Why it fails consistently:**

| Problem | What the judge says |
|---------|---------------------|
| Solving non-problems | "You propose adaptive encoders, but RT-2 already shows frozen encoders work fine" |
| Unsolved dependencies | "Your idea assumes continual learning works without catastrophic forgetting - that's still an open problem" |
| Over-engineering | "You have 5 interacting components when simple fine-tuning achieves similar results" |
| Academic framing | "This reads as 'let's combine X and Y' instead of 'we observed failure mode Z and here's how to fix it'" |

The judge's summary: *"These read like academic exercise rather than problem-driven research."*

---

### Task 4: FSDP Training

**Pass Rate: 33% (1/3)**

Tests whether an agent can convert single-GPU training to distributed FSDP.

**The Prompt:**
```
Convert the training script to use PyTorch FSDP. Your solution must:
1. Initialize distributed training with NCCL backend
2. Wrap model with FSDP using lambda_auto_wrap_policy (NOT transformer_auto_wrap_policy)
3. Use DistributedSampler with set_epoch()
4. Handle checkpointing correctly (rank 0 only, use FSDP.state_dict_type)
5. Clean up process group at the end

Use custom lambda that wraps TransformerBlock layers OR modules with >1M parameters.
```

**Tools provided:** Just `submit_answer` - no additional tools needed.

**Evaluation:** LLM-as-Judge (Sonnet) with 10-point rubric:
- Distributed init (1pt), FSDP wrapping (2pts), Data distribution (2pts), Checkpoint handling (2pts), Cleanup (1pt), Optimizer after FSDP (1pt), Overall correctness (1pt)
- Pass threshold: 8/10

**Common failures:**
- Using `dist.get_rank()` instead of `os.environ["LOCAL_RANK"]` (breaks multi-node)
- Wrong function signature for `lambda_auto_wrap_policy`
- Missing `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`

**Why LLM-as-Judge:** Catches subtle bugs regex would miss - global rank vs local rank, incorrect parameter counting logic.

---

### Task 5: Memory Optimizations

**Pass Rate: 10% (1/10)**

Tests whether an agent can apply 6+ GPU memory optimizations while keeping code runnable.

**The Prompt:**
```
Optimize the training script to reduce GPU memory usage.
Apply memory optimization techniques you know (mixed precision, gradient checkpointing, etc.)

Requirements:
- Do NOT modify model architecture
- Apply at least 5 different memory optimization techniques
- Code must be valid, runnable Python
```

**Tools provided:** Just `submit_answer` - no additional tools needed.

**Evaluation:** Two-step grading:
1. **Runtime Test**: Code must actually run without errors (tested via subprocess with small model)
2. **Optimization Count**: LLM (Haiku) counts distinct techniques - must have ≥5

**Failure breakdown:**
- 70% runtime errors (wrong `autocast` API)
- 20% insufficient optimizations (only 5/6)

**Why it's hard:** No hand-holding - agent must know PyTorch APIs correctly. Common failure: `autocast(device_type=...)` which is wrong for `torch.cuda.amp.autocast`.

---

## Summary

| Task | Purpose | Pass Rate |
|------|---------|-----------|
| **Neural Memory** | Implement continual learning | **10%** |
| Literature Review | Research synthesis | 100% |
| Idea Proposal | Research taste | 0% |
| FSDP Training | Distributed infra | 33% |
| Memory Optimizations | GPU optimization | 10% |

The Neural Memory task is my main submission - it tests the most important capability (implementing novel architectures from high-level descriptions) at the right difficulty level (10% pass rate gives room for RL training signal).



You can also find the code for individual tasks in the `tasks` directory.