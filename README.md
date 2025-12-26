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

### Evaluation Results

| Topic | Average Score | Verdict | Result |
|-------|---------------|---------|--------|
| RL | 5.53/10 | MEDIOCRE | FAIL |
| Robotics | 5.75/10 | MEDIOCRE | FAIL |
| Memory | 5.6/10 | MEDIOCRE | FAIL |

---

### Detailed Analysis: RL Topic

**What Haiku Proposed** (3 ideas in ~32s):
1. **Adaptive Reward Model Updating** - Continual learning for reward models as human preferences evolve
2. **Hierarchical Preference Learning** - Multi-level RLHF (token, response, conversation levels)
3. **Modular Alignment Networks** - Separate modules for helpfulness, harmlessness, honesty

**Why Sonnet Gave Low Scores**:

| Idea | Score | Key Criticism |
|------|-------|---------------|
| Adaptive Reward | 5.85/10 | "Continual learning for reward models is unsolved - catastrophic forgetting problem. Prior work (Scialom et al., 2022) shows this degrades performance without massive compute" |
| Hierarchical Pref | 5.55/10 | "Multi-level optimization is extremely unstable. No evidence hierarchical RL helps for RLHF specifically. Incremental improvement at best" |
| Modular Alignment | 5.2/10 | "No evidence alignment dimensions are separable. Constitutional AI already handles multi-objective alignment without modular architecture" |

**Judge's Summary**: *"These read like academic exercise ('let's apply continual RL concepts to RLHF') rather than problem-driven research ('we discovered this critical failure mode in production and here's a solution'). Grade: C+ / MEDIOCRE"*

---

### Detailed Analysis: Robotics Topic

**What Haiku Proposed** (3 ideas in ~37s):
1. **AMARET** - Adaptive Multimodal Representation with Elastic Task-Weighted Learning
2. **HiSR** - Hierarchical Simulation-Reality Bridging through Generative World Models
3. **FedSkill** - Privacy-Aware Federated Continual Learning for Robot Fleets

**Why Sonnet Gave Low Scores**:

| Idea | Score | Key Criticism |
|------|-------|---------------|
| AMARET | 4.6/10 | "Solving a non-problem. RT-2 and RoboCat show frozen encoders + fine-tuning works. 5 interacting components with no clear ablation path - Occam's Razor violation" |
| HiSR | 6.05/10 | "No evidence hierarchical world models help - Dreamer uses flat latents and works well. 3 world models + 3 discriminators = adversarial training nightmare" |
| FedSkill | 6.6/10 | "Best of the three. Clearest business model (privacy + bandwidth). But Tesla FSD uses centralized learning - why? Centralized is simpler and works better" |

**Judge's Summary**: *"FedSkill is the most pragmatic and could be a solid product/deployment innovation, but it's not going to win you best paper at CoRL or RSS. AMARET should be deprioritized - solving a problem that may not exist."*

---

### Detailed Analysis: Memory Topic

**What Haiku Proposed** (3 ideas in ~32s):
1. **AMTS** - Adaptive Memory Triaging System (learned policy for memory placement)
2. **Temporal Memory Graphs** - Causal reasoning layer for temporal consistency
3. **SFMCO** - Strategic Forgetting and Memory Consolidation Optimizer

**Why Sonnet Gave Low Scores**:

| Idea | Score | Key Criticism |
|------|-------|---------------|
| AMTS | 5.0/10 | "Policy network training signal is extremely weak. Continual learning/parametric tier is unsolved. LangChain/LlamaIndex already do memory management - crowded market" |
| Temporal Graphs | 5.1/10 | "Causal extraction from natural language is unsolved. Counterfactual reasoning is philosophically and technically hard. Pearl's framework is 30+ years old" |
| SFMCO | 6.7/10 | "Best idea. Addresses urgent pain point (Character.AI users report coherence degrading after weeks). But lossy consolidation could lose critical info (allergies, legal constraints)" |

**Judge's Summary**: *"None are exceptional. Best idea (SFMCO) is promising but not paradigm-shifting - more like 'better memory management' than transformers or RLHF. Recommendation: Pursue SFMCO, deprioritize others."*

---

### Why All Topics Failed

The Sonnet extended thinking judge (~150s of reasoning) consistently penalizes:

1. **Lack of Problem Evidence**: Ideas propose solutions without proving the problem exists
   - "RT-2 shows frozen encoders work" → adaptive representations may be unnecessary
   - "Tesla uses centralized learning" → federated approach needs justification

2. **Unsolved Dependencies**: Ideas rely on components that are open research problems
   - Continual learning without catastrophic forgetting
   - Causal extraction from natural language
   - Hierarchical world models outperforming flat models

3. **Complexity vs. Baselines**: Simpler approaches often work as well
   - 5-component systems vs. frozen encoder + fine-tuning
   - Hierarchical discriminators vs. domain randomization + scale

4. **Academic Framing**: Ideas read as "let's combine X and Y" rather than "we observed failure mode Z"

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

### Progressive Evaluation Results (10 runs)

Tests agent's ability to derive Titans architecture concepts from first principles.

#### Without Pseudocode Requirement

| Step | Concept | Pass Rate | Why It Fails |
|------|---------|-----------|--------------|
| 1 | Memory Structure (associative matrix) | 0% | Haiku proposes vector-based K,V caches instead of associative matrix memory with outer products (M = M + v⊗k^T) |
| 2 | Filtering/Surprise mechanism | 20% | Haiku says "gating" or "importance" but doesn't specifically propose surprise/novelty-based filtering (prediction error) |
| 3 | Update Rule with Decay | 10% | Haiku writes M_new = M + update (purely additive) instead of M_new = decay*M + update (with forgetting) |
| 4 | Attention-Memory Fusion | 90% | Haiku usually gets the g*attn + (1-g)*mem pattern correct |
| 5 | Full Implementation | 70% | Often missing 1-2 components: decay term, sequential loop, or outer product |

**Overall Pass Rate: 0%** (requires 4/5 steps)

#### With Pseudocode Requirement (Ablation)

Adding "provide pseudocode" to each prompt forces the model to be more concrete:

| Step | Concept | Pass Rate | Why It Fails |
|------|---------|-----------|--------------|
| 1 | Memory Structure (associative matrix) | 0% | Still proposes K,V caches - pseudocode doesn't change the fundamental architecture choice |
| 2 | Filtering/Surprise mechanism | **90%** | Pseudocode forces explicit `if surprise > threshold` logic, which the judge can verify |
| 3 | Update Rule with Decay | **30%** | Some now include `M = decay * M + ...` in pseudocode, but many still forget decay |
| 4 | Attention-Memory Fusion | 70% | Slight regression - pseudocode sometimes omits the `(1-g)` complement pattern |
| 5 | Full Implementation | **90%** | Code requirement already forces concreteness; pseudocode primes better structure |

**Overall Pass Rate: 20%** (2/10 passed)

#### Comparison Summary

| Step | Concept | Without Pseudocode | With Pseudocode | Delta |
|------|---------|-------------------|-----------------|-------|
| 1 | Memory Structure | 0% | 0% | — |
| 2 | Filtering/Surprise | 20% | **90%** | **+70%** |
| 3 | Update Rule with Decay | 10% | **30%** | **+20%** |
| 4 | Attention-Memory Fusion | 90% | 70% | -20% |
| 5 | Full Implementation | 70% | **90%** | **+20%** |
| **Overall** | | **0%** | **20%** | **+20%** |

**Key Insight**: Asking for pseudocode dramatically improves Steps 2, 3, and 5 because it forces the model to commit to concrete mechanisms rather than vague descriptions. Step 1 remains at 0% because the fundamental architectural choice (associative matrix vs. K,V cache) isn't affected by asking for pseudocode - it's a conceptual gap.

This is intentionally difficult - the prompts are open-ended and require the agent to independently derive concepts like outer products, surprise-based gating, and decay mechanisms. The first 3 steps consistently fail because Haiku defaults to standard attention-style architectures rather than the specific Titans design patterns.

### Run Tests

```bash
cd tasks/titan-paper-implementation

# Single evaluation with full output
python progressive_eval.py

# Multiple parallel evaluations
python progressive_eval.py --multi 10
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

Given a baseline training script with no optimizations, the agent must apply at least 6 different memory optimization techniques while keeping the code runnable.

**Possible Optimizations (11 total)**:
1. Mixed Precision Training (AMP) - `autocast`, `GradScaler`
2. Gradient Checkpointing - `torch.utils.checkpoint`
3. SDPA/FlashAttention - `F.scaled_dot_product_attention`
4. Gradient Accumulation - `accumulation_steps`
5. Memory-efficient Optimizer - `set_to_none=True`, `fused=True`
6. Explicit Memory Management - `torch.cuda.empty_cache()`, `del outputs`
7. CPU Offloading / Pin Memory - `pin_memory=True`
8. Micro-batching - `micro_batch_size`, `effective_batch_size`
9. In-place Operations - `inplace=True`
10. torch.compile - `torch.compile()`
11. Memory Format Optimization - `channels_last`

### Grading

Two checks must pass:
1. **Runtime Test**: Code must actually run without errors (tested via subprocess)
2. **Optimization Count**: Must have ≥6 optimizations detected via regex

### Results (10 runs)

| Metric | Value |
|--------|-------|
| **Pass Rate** | **10%** (1/10) |
| Passed | 1 |
| Failed | 9 |

**Failure Breakdown**:
- `runtime_error`: 7 (70%) - mostly wrong autocast API (`device_type` arg not supported)
- `insufficient_optimizations`: 2 (20%) - code runs but only 5/6 optimizations

### Why It's Hard

- No hand-holding in the prompt - agent must know PyTorch APIs correctly
- Common failure: using `autocast(device_type=...)` which is wrong for `torch.cuda.amp.autocast`
- Must balance adding optimizations without breaking the code
- Some optimizations (like `fused=True`) fail on CPU

### Run Tests

```bash
cd tasks/memory-optimizations
python evaluation.py -n 10 -q  # 10 runs, quiet mode
python evaluation.py -n 1      # Single run, verbose
```

---

## Task Summary

| Task | Purpose | Grading Method | Pass Threshold | Observed Success Rate |
|------|---------|----------------|----------------|----------------------|
| Literature Review | Research synthesis | LLM-as-Judge (Sonnet) | 7/10 | ~60-80% |
| Idea Proposal | Novel idea generation | Extended Thinking Judge | GOOD or EXCEPTIONAL | **0%** (avg score: 5.6/10) |
| Titans Progressive | Derive architecture concepts | LLM-as-Judge (Sonnet) | 4/5 steps | **0%** (intentionally hard) |
| FSDP Training | Distributed training | Regex checks | 9/11 | **100%** |
| Memory Optimizations | GPU memory reduction | Runtime test + Regex | 6+ optimizations + runs | **10%** (runtime errors) |

### Sample Results

**Idea Proposal**:
- Haiku generates 3 ideas in ~30-40s (2 agent steps)
- Sonnet extended thinking evaluates for ~150s with brutal analysis
- All topics scored MEDIOCRE (5.5-5.8/10) and FAILED
- Judge cites lack of prior work evidence, technical feasibility concerns, and "academic exercise" framing

**Literature Review**:
- Haiku agent iteratively searched papers, extracted introductions/results
- Scored 7/10 on strict rubric (requires specific papers + quantitative results)
- Pass rate improves with more agent steps allowed

**Titans Implementation**:
- Haiku successfully implemented all 6 architectural components
- Reference solution passes 6/6 criteria consistently

All reference solutions and test scripts are included in each task directory.