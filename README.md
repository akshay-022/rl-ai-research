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

### Evaluation Results

**Result: PASS (8/10)** - Threshold: 7/10

| Approach | Topics | Results | Score |
|----------|--------|---------|-------|
| 1. Extended Context | PASS | PASS | 2/2 |
| 2. RAG/Retrieval | PASS | PASS | 2/2 |
| 3. Summarization | PASS | PASS | 2/2 |
| 4. Memory Architectures | PASS | PASS | 2/2 |

#### Key Findings

- Agent produced **28K-character review** with 20+ papers and quantitative results
- Successfully extracted results using LLM-based PDF section extraction (`claude-haiku-4-5`)
- Papers found: LongLoRA, Infini-attention, InstructRetro, LLMLingua-2, MemGPT, HiAgent, etc.

#### Sample Results Extracted

| Paper | Key Metric |
|-------|------------|
| LongLoRA | Extends Llama2 7B from 4k→100k context |
| Infini-attention | 96-100% accuracy at 1M token passkey retrieval |
| InstructRetro | 7-16% improvement over GPT-43B on QA tasks |
| LLMLingua-2 | 3-6x compression with 79% EM on GSM8K |
| MemGPT | 92.5% vs 32.1% baseline on DMR benchmark |


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

## Neural Memory Implementation Task

Tests whether an agent can implement a neural long-term memory module from biological hints, getting all the technical details right.

### The Challenge

Given a simple Transformer and hints about how biological memory works, the agent must add a Neural Long-Term Memory module with:

1. **NeuralMemory Class** - A proper nn.Module with memory logic
2. **Surprise Gate with Sigmoid** - Learned metric controlling memory writes
3. **Learnable Decay Parameter** - nn.Parameter for memory fade rate
4. **Outer Product for Association** - V ⊗ K^T for key-value binding
5. **Recurrent Memory Update** - Per-timestep loop updating memory state
6. **Memory-Attention Gating** - Learned fusion of attention and memory outputs

### Grading

- **Execution Test**: Code must run and produce correct output shape
- **LLM-as-Judge** (Sonnet 4.5) evaluates code against 6 strict criteria
- **Pass threshold**: 6/6 (all criteria must pass)
- Reference solution provided in `titans_solution.py`

### Results (Sonnet 4.5, 10 runs)

**Pass Rate: 10% (1/10)** ✓ In target range

The task provides hints about biological memory (surprise, decay, associations, blending) and the outer product formula, but does NOT explicitly tell the model about surprise gates or fusion formulas.

#### What Models Consistently Get Right (4/6)

| Criterion | Pass Rate | Why It Passes |
|-----------|-----------|---------------|
| NeuralMemory Class | 100% | Clear requirement in prompt |
| Learnable Decay | ~90% | "Make parameters learnable" hint works |
| Outer Product | ~90% | Explicit formula provided: `M += V ⊗ K^T` |
| Recurrent Update | ~90% | "Process tokens one at a time" is explicit |

#### What Models Consistently Get Wrong (Differentiators)

| Criterion | Pass Rate | Why It Fails |
|-----------|-----------|--------------|
| Surprise Gate | ~20% | Models implement generic learned gates (e.g., `write_gate = sigmoid(Linear(x))`), not prediction-error based surprise (comparing input to what memory predicts). The prompt hints at "surprising or novel information" but doesn't say HOW to compute surprise. |
| Memory-Attention Gating | ~20% | Models use `x + g * mem` (residual) or `concat(attn, mem) → Linear` instead of the required complementary gating: `g * attn + (1-g) * mem`. The `(1-g)` pattern is the key insight models miss. |

#### Example Failure Patterns

**Surprise Gate Failures:**
```python
# What models write (FAILS):
write_gate = torch.sigmoid(self.gate_proj(x_t))  # Just a learned gate

# What's required (PASSES):
retrieved = M @ query  # What memory predicts
surprise = torch.sigmoid(self.surprise_net(x_t - retrieved))  # Prediction error
```

**Memory-Attention Gating Failures:**
```python
# What models write (FAILS):
output = x_t + gate * memory_out  # Residual addition
output = self.blend(torch.cat([attn, mem], dim=-1))  # Concatenation

# What's required (PASSES):
g = torch.sigmoid(self.gate(torch.cat([attn, mem], dim=-1)))
output = g * attn + (1 - g) * mem  # Complementary interpolation
```

#### The One That Passed (Run 5)

The successful submission implemented:
- `novelty_net` comparing input with retrieved memory
- Proper `gate_output` with `g * mem + (1-g) * x` formula
- All other criteria correctly

### Run Tests

```bash
cd tasks/titan-paper-implementation

# Single run (Haiku by default)
python combined_task.py

# Single run with Sonnet
python combined_task.py claude-sonnet-4-5-20250929

# 10 parallel runs with Sonnet
python combined_task.py --multi 10 claude-sonnet-4-5-20250929
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

### Grading

Uses **LLM-as-a-judge** (Claude Sonnet) with a 10-point rubric:

| Component | Points | What's Checked |
|-----------|--------|----------------|
| Distributed Initialization | 1 | `init_process_group()` + LOCAL_RANK handling |
| FSDP Model Wrapping | 2 | FSDP import + `lambda_auto_wrap_policy` with custom lambda |
| Data Distribution | 2 | `DistributedSampler` + `set_epoch()` call |
| Checkpoint Handling | 2 | Rank 0 only + `FSDP.state_dict_type` with `FullStateDictConfig` |
| Cleanup | 1 | `destroy_process_group()` |
| Optimizer Creation | 1 | Created AFTER FSDP wrap |
| Overall Correctness | 1 | Would run correctly with torchrun |

**Pass threshold**: 8/10 points

### Evaluation Results (3 runs)

| Run | Score | Result | Key Issues |
|-----|-------|--------|------------|
| 1 | 9/10 | **PASS** ✓ | Only missing `FullStateDictConfig` |
| 2 | 5/10 | FAIL | Broken wrap_policy, wrong LOCAL_RANK, missing FullStateDictConfig |
| 3 | 5/10 | FAIL | Wrong LOCAL_RANK (uses global rank), broken param counting in wrap policy |

**Pass Rate: 33.3% (1/3)**

### Common Failure Modes

1. **LOCAL_RANK handling** - Haiku often uses `dist.get_rank()` instead of `os.environ["LOCAL_RANK"]` to set CUDA device. This works for single-node but breaks multi-node training.

2. **Broken wrap_policy implementations** - The custom lambda function is often incorrectly implemented:
   - Wrong function signature for `lambda_auto_wrap_policy`
   - Counts all parameters recursively instead of just the module's own parameters

3. **Missing `FullStateDictConfig`** - Most attempts miss including `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` in the checkpoint saving context manager.

### Why LLM-as-Judge (not regex)

The LLM evaluator catches subtle bugs that regex would miss:
- Using global rank vs local rank (both contain "rank" but one is wrong)
- Incorrect parameter counting in wrap policy (logic error, not syntax)
- Wrong function signatures that would cause runtime errors

### Run Tests

```bash
cd tasks/fsdp-training

# Run agent evaluation (Haiku converts code, Sonnet grades)
python evaluation.py -n 3      # 3 runs
python evaluation.py -n 1 -q   # Single run, quiet

# Unit tests (no API calls)
SKIP_LLM_TESTS=1 python -m pytest test_grading.py -v
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
| Literature Review | Research synthesis | LLM-as-Judge (Sonnet) | 7/10 | **100%** (8/10) |
| Idea Proposal | Novel idea generation | Extended Thinking Judge | GOOD or EXCEPTIONAL | **0%** (avg score: 5.6/10) |
| Neural Memory | Implement neural memory | Execution + LLM-as-Judge | 6/6 criteria | **10%** (Sonnet 4.5) |
| FSDP Training | Distributed training | LLM-as-Judge (Sonnet) | 8/10 | **33%** (subtle bugs) |
| Memory Optimizations | GPU memory reduction | Runtime test + Regex | 6+ optimizations + runs | **10%** (runtime errors) |
