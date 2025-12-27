# Final Documentation

I was asked to create RL environments for an automated AI researcher. I found this assignment quite enjoyable, this is relevant to making AI disover new science to push humanity forward and I find that very exciting.

Instead of just trying to implement a single coding task in pytorch that the model fails at, I wanted to run many experiments along multiple dimensions of research competency to see how good AI is at it currently.

Let me first talk about my main submission and then all the other experiments I ran.

---

## Main Submission: Neural Memory Implementation Task

**Pass Rate: 10% (Sonnet 4.5)**

This is the biggest unhobbling needed to make AI superhuman, all current RL is cope to not have algorithms to do this well.

Moreover, I believe these continual learning models must have memory and it MUST be in the model weights, and not simply in stored context. The model weights are much higher dimensional, that is also the way our brain works (memory is stored in synapse weights).

### The Challenge

Given a simple transformer and hints about how biological memory works, the agent must implement a Neural Long-Term Memory module with 6 specific components:

1. **NeuralMemory Class** - A proper nn.Module with memory logic
2. **Surprise Gate with Sigmoid** - Learned metric controlling memory writes
3. **Learnable Decay Parameter** - nn.Parameter for memory fade rate
4. **Outer Product for Association** - V ⊗ K^T for key-value binding
5. **Recurrent Memory Update** - Per-timestep loop updating memory state
6. **Memory-Attention Gating** - Learned fusion: `g * attn + (1-g) * mem`

### Why This Task Matters

The prompt gives biological hints (surprising info, fading memories, associations) and the outer product formula, but does NOT explicitly tell the model about surprise gates or fusion formulas. The model must derive these from first principles.

This was a task to give a model a research direction and ask it to implement the experiment while getting all the pesky details right. 

### Results

Models consistently get 4 out of 6 criteria right: the NeuralMemory class (explicitly required), learnable decay (prompted with "make parameters learnable"), outer product (formula given: `M += V ⊗ K^T`), and recurrent update ("process tokens one at a time").

**Where models fail:**

The **Surprise Gate** (~80% failure rate) is where most models fall down. They implement generic learned gates like `write_gate = sigmoid(Linear(x))` instead of computing prediction-error based surprise by comparing input to what memory predicts (e.g., `surprise = sigmoid(net(x - retrieved))`). The prompt hints at "surprising or novel information" but doesn't say HOW to compute surprise.

The **Memory-Attention Gating** (~80% failure rate) is the other differentiator. Models use residual patterns like `output = x + gate * mem` or concatenation `output = Linear(concat(attn, mem))` instead of the complementary interpolation `g * attn + (1-g) * mem`. The `(1-g)` pattern is the key insight models miss.

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

---

## Other Experiments

I wanted to test the limits of AI in how well it can do literature survey, how good the research directions it proposes are, and how well it can do infra optimizations to run experiments.

### Task 2: Literature Review

**Pass Rate: 100% (8/10 score)**

Tests how comprehensively an LLM can survey a research field.

**Tools provided:**
- `web_search(query)` - Search arXiv/Semantic Scholar
- `get_paper_introduction(paper_id)` - Extract intro (contains authors' lit review)
- `get_paper_results(paper_id)` - Extract quantitative results

**Evaluation criteria:**
- Coverage of major approaches (must cite specific papers + numbers)
- Seminal works cited with results
- Quality of synthesis (comparisons, trade-offs)

**Result:** Agent produced 28K-character review with 20+ papers. Successfully found LongLoRA (4k→100k context), Infini-attention (96-100% at 1M tokens), MemGPT (92.5% vs 32.1% baseline).

---

### Task 3: Idea Proposal

**Pass Rate: 0% (avg 5.6/10)**

Tests research taste - can the model propose ideas that are both novel AND likely to succeed?

**Evaluation:** Sonnet extended thinking (~150s reasoning) judges each idea on:
- Business Viability (30%)
- Potential Upside (25%)
- Likelihood of Success (30%)
- Novelty (15%)

**Why it fails consistently:**

| Problem | Example |
|---------|---------|
| Lack of Problem Evidence | "RT-2 shows frozen encoders work" → adaptive representations unnecessary |
| Unsolved Dependencies | Relies on continual learning without catastrophic forgetting |
| Complexity vs Baselines | 5-component systems vs simple fine-tuning |
| Academic Framing | "Let's combine X and Y" vs "we observed failure mode Z" |

The judge's summary: *"These read like academic exercise rather than problem-driven research."*

---

### Task 4: FSDP Training

**Pass Rate: 33% (1/3)**

Tests whether an agent can convert single-GPU training to distributed FSDP.

**Common failures:**
- Using `dist.get_rank()` instead of `os.environ["LOCAL_RANK"]` (breaks multi-node)
- Wrong function signature for `lambda_auto_wrap_policy`
- Missing `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`

**Why LLM-as-Judge:** Catches subtle bugs regex would miss - global rank vs local rank, incorrect parameter counting logic.

---

### Task 5: Memory Optimizations

**Pass Rate: 10% (1/10)**

Tests whether an agent can apply 6+ GPU memory optimizations while keeping code runnable.

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
