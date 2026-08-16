# Deployment and Capstone — Full Course Module

## Module Overview
This module takes learners from working prototypes to production-grade GenAI systems. The focus is on performance optimization, reliability engineering, observability, and human-centered design. The capstone ties everything together by building a language model from scratch and shipping it.

## Target Audience
- Developers and technical professionals
- Completed Months 1–5 (full stack of GenAI development)

## Learning Objectives
By the end of this module, learners will be able to:
- Optimize inference latency and cost through caching, batching, and routing
- Set up monitoring, logging, tracing, and alerting for GenAI systems
- Design trustworthy user experiences with human-in-the-loop controls
- Deploy a custom-trained model with a REST API and web UI

---

## Prerequisites
- Months 1–5: Full GenAI stack (architecture, agents, RAG, fine-tuning, multimodal, evaluation, safety)
- Python 3.10+
- A trained model from the capstone project (or a fine-tuned model from Month 3)
- Cloud account (optional — local deployment is sufficient)

---

## Module Structure

| Module | Topic | Lab |
|--------|-------|-----|
| 6.1 | Performance and Cost | Cost/latency optimization experiment |
| 6.2 | Observability and Reliability | Monitoring dashboard for your system |
| 6.3 | UX and Human-in-the-Loop | Usability test and iteration |
| Capstone | Train + ship a model from scratch | End-to-end project |

---

# Module 6.1: Performance and Cost

## Core Concepts

### 1. The Latency Stack

Every GenAI response goes through these layers. Optimizing the wrong one wastes effort.

```
User request
    |
    v
1. Network latency     — CDN, geographic routing, connection reuse
    |
    v
2. Auth / preprocessing — Token counting, guardrail checks, input validation
    |
    v
3. Model inference     — Forward pass + decoding (the dominant cost)
    |
    v
4. Post-processing     — Output validation, formatting, safety checks
    |
    v
5. Response delivery   — Streaming vs buffered, compression
```

**Where to focus:** Inference (layer 3) is 80-90% of total latency for most systems. Optimize there first.

### 2. Inference Optimization Techniques

**Quantization**

Reduce model precision to speed up inference and reduce memory.

| Precision | Bits per param | Memory (10M model) | Speedup vs FP32 |
|-----------|---------------|-------------------|-----------------|
| FP32 | 32 | 40 MB | 1x |
| FP16 / BF16 | 16 | 20 MB | ~2x |
| INT8 | 8 | 10 MB | ~3-4x |
| INT4 | 4 | 5 MB | ~4-6x |

```python
# torch.compile — free speedup
model = torch.compile(model, mode="reduce-overhead")

# FP16 inference
with torch.amp.autocast(device_type="cuda"):
    logits = model(x)
```

**KV Caching**

During autoregressive generation, cache the key and value tensors from previous steps so you don't recompute them.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.kv_cache = None  # (B, n_kv_heads, T_cache, head_dim)

    def forward(self, x, use_cache=True):
        # First token: compute full Q, K, V
        # Subsequent tokens: compute only new Q, K, V, append to cache
        # Attention over full cache (past + new tokens)
        pass
```

Without KV cache: O(T^2) per step  
With KV cache: O(T) per step

**Continuous Batching**

Process multiple requests in the same forward pass. Instead of waiting for one generation to finish, batch new requests into the slots that have finished.

```
Time ──────────────────────────────────────────────>
      ┌─────┐
      | Req1 | (finished early)
      └─────┘    ┌─────┐
                 | Req2|
                 └─────┘    ┌─────┐
                            | Req3|
                            └─────┘
Naive: requests wait for previous to finish

      ┌─────┐────┐────┐
      | Req1|Req2|Req3|  (all processed together)
      └─────┘────┘────┘
Continuous batching: fill empty slots immediately
```

### 3. Cost Optimization

**Token Budgets**

Track and enforce limits per request, per user, per day.

| Strategy | How It Works | Savings |
|----------|-------------|---------|
| Max tokens cap | Hard limit on output length | Prevents runaway costs |
| Prompt compression | Reduce input tokens via summarization | 40-60% fewer input tokens |
| Caching | Reuse responses for identical inputs | 100% on cache hits |
| Model routing | Simple queries -> small/cheap model | 50-80% cost reduction |

**Model Routing**

```
Request
    |
    v
Classifier (small model)
    |
    +-- Simple task --> Small/cheap model (e.g., GPT-4o-mini)
    |
    +-- Complex task -> Large/expensive model (e.g., GPT-4o)
    |
    +-- Domain task  --> Your custom model
```

### 4. Caching Strategies

| Cache Type | Key | Value | Eviction |
|-----------|-----|-------|----------|
| Exact match | Input text hash | Output | LRU, TTL |
| Semantic cache | Input embedding | Output | Similarity threshold |
| Prefix cache | Prompt prefix | KV cache | TTL |

---

## Lab: Cost/Latency Optimization Experiment

### Goal
Take a model (your capstone model or fine-tuned model from Month 3) and optimize it across three dimensions: latency, memory, and cost.

### Steps
1. **Baseline measurement**  
   Measure current: latency per token, peak memory, total inference time for 100 generations.

2. **Apply optimizations**  
   - `torch.compile` with different modes
   - FP16 inference
   - KV caching (implement if not already)
   - INT8 quantization (torch.quantization or bitsandbytes)

3. **Measure improvements**  
   Create a comparison table:

| Optimization | Latency (100 tok) | Peak Memory | Throughput (tok/s) |
|-------------|-------------------|-------------|-------------------|
| Baseline (FP32) | 450ms | 1.2 GB | 220 |
| + compile | 280ms | 1.2 GB | 357 |
| + FP16 | 190ms | 600 MB | 526 |
| + KV cache | 120ms | 650 MB | 833 |
| + INT8 | 95ms | 350 MB | 1052 |

### Deliverable
`optimization_report.md` with before/after measurements and analysis of which optimization gave the best ROI.

---

## Exercises

1. **Bottleneck Identification**  
   Your system has 500ms total latency: 50ms network, 30ms preprocessing, 400ms inference, 20ms post-processing. Where do you optimize first, and what technique do you use?

2. **Cost Modeling**  
   Your app does 10K requests/day, each using 500 input tokens and 200 output tokens via GPT-4o ($0.03/$0.06 per 1K tokens). What's the monthly cost? How much would model routing save if 70% of requests can use GPT-4o-mini ($0.002/$0.002)?

---

## Assignment (Graded)

### Task
Optimize a model's inference performance and produce a cost/latency analysis report.

### Requirements
- Apply at least 3 optimization techniques
- Measure and document each optimization's impact
- Build a cost model for your use case at 1K, 10K, and 100K requests/day

### Deliverable
`optimization_report.md` with baseline, optimizations, measurements, and cost projections.

### Rubric (100 points)
- **Baseline measurement (20 points)**: Clear, reproducible benchmarks
- **Optimization breadth (30 points)**: At least 3 techniques, correctly applied
- **Analysis depth (30 points)**: Understanding of tradeoffs, not just numbers
- **Cost model (20 points)**: Realistic projections with assumptions stated

---

# Module 6.2: Observability and Reliability

## Core Concepts

### 1. The Three Pillars of Observability

**Logging** — Record discrete events

```json
{
  "timestamp": "2026-07-29T12:00:00Z",
  "level": "WARN",
  "request_id": "req_abc123",
  "message": "High latency detected",
  "latency_ms": 3200,
  "model": "gpt-4o",
  "user_id": "user_789",
  "input_tokens": 450,
  "output_tokens": 1200
}
```

**Metrics** — Aggregated measurements over time

- Latency: p50, p95, p99
- Error rate: 4xx, 5xx, timeout
- Throughput: requests per second
- Token usage: input/output tokens per hour
- Cost: $ per day, per user, per feature

**Tracing** — End-to-end request flow across services

```
Frontend -> API Gateway -> Auth -> Model Server -> RAG Pipeline -> LLM
    |            |          |          |               |           |
    +------------+----------+----------+---------------+-----------+
                              Trace ID: trc_456
```

### 2. Key Metrics for GenAI Systems

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| Latency | Time to first token | <500ms | >2s p95 |
| Latency | Time to last token | <5s | >15s p95 |
| Quality | Thumbs up rate | >80% | <60% |
| Quality | Hallucination flag rate | <1% | >5% |
| Safety | Content flag rate | <0.5% | >2% |
| Cost | Cost per request | <$0.01 | >$0.05 |
| Reliability | Error rate | <0.1% | >1% |
| Reliability | Uptime | 99.9% | <99.5% |

### 3. Alerting

**Alert types:**
- **Page-worthy**: Service down, p99 latency > 10s, error rate > 5%
- **Ticket-worthy**: p95 latency creeping up, cost spike, quality dip
- **Dashboard-worthy**: Everything else (visible but no immediate action)

**Incident response:**
```
Detect -> Triage -> Mitigate -> Resolve -> Post-mortem
  |          |          |           |            |
  Alert     Assess    Fix/rollback Confirm     Document
            severity                        what went wrong
```

---

## Lab: Monitoring Dashboard for Your System

### Goal
Set up a monitoring dashboard for your deployed model that tracks latency, error rate, token usage, and cost.

### Steps
1. **Instrument your API**  
   Add logging and metrics collection to your FastAPI server.

2. **Set up a dashboard**  
   Use Prometheus + Grafana locally, or a lightweight alternative.

3. **Define SLOs**  
   - Latency SLO: 95% of requests complete under 2s  
   - Error SLO: 99.9% of requests return 200  
   - Cost SLO: average cost per request under $0.01

4. **Configure alerts**  
   - Warning: p95 latency > 1.5s for 5 minutes
   - Critical: error rate > 2% for 1 minute

### Deliverable
`slo_report.md` with SLO definitions, dashboard screenshot/description, and alert rules.

---

## Exercises

1. **SLO Design**  
   Define SLOs for three systems: (a) real-time chatbot, (b) batch document processor, (c) code generation assistant.

2. **Incident Response**  
   Your monitoring shows a sudden spike in p99 latency from 2s to 12s. Write the first 3 actions you take.

---

## Assignment (Graded)

### Task
Define SLOs for a GenAI system, set up monitoring, and write an incident response plan.

### Requirements
- 3 SLOs with measurement methods and targets
- Dashboard design (implemented or mocked up)
- Alert rules with thresholds and escalation paths
- Incident response playbook for 3 scenarios

### Deliverable
- `slo_report.md`
- `incident_playbook.md`

### Rubric (100 points)
- **SLO quality (30 points)**: Meaningful, measurable, user-focused
- **Dashboard (25 points)**: Covers health, satisfaction, and safety
- **Alert design (20 points)**: Appropriate thresholds, no alert fatigue
- **Playbook (25 points)**: Practical, specific steps, clear ownership

---

# Module 6.3: UX and Human-in-the-Loop

## Core Concepts

### 1. Designing for Trust

Users trust GenAI systems when they are:
- **Predictable**: Same input produces similar output
- **Transparent**: The system communicates its uncertainty
- **Controllable**: Users can steer, correct, or reject outputs
- **Accountable**: Every output can be traced and audited

### 2. UX Patterns for GenAI

| Pattern | Description | Example |
|---------|-------------|---------|
| Progressive disclosure | Show simple output first, allow drill-down | Summary first, then details on click |
| Confidence indicators | Show how sure the system is | "High confidence (92%)" badge |
| Source citations | Link outputs to their sources | Numbered references in RAG output |
| Edit and resubmit | Allow users to edit the output | "Tweak this" button |
| Undo / revert | Roll back AI actions | "Undo" on generated content |

### 3. Human-in-the-Loop (HITL)

**When to require human review:**
- High-stakes decisions (medical, legal, financial)
- First N outputs from a new model version
- Outputs below a confidence threshold
- Edge cases the system hasn't seen before

```
Request
    |
    v
Model generates output + confidence score
    |
    v
Score > threshold? --> Auto-approve
    |
    No
    |
    v
Send to human reviewer queue
    |
    v
Human reviews, edits, approves/rejects
    |
    v
Feedback logged -> used for improvement
```

### 4. Audit Trails

Every GenAI interaction should be logged for audit:

```json
{
  "timestamp": "2026-07-29T12:00:00Z",
  "user_id": "user_789",
  "input": "What is the refund policy?",
  "output": "Our refund policy allows returns within 30 days...",
  "model": "gpt-4o",
  "model_version": "2026-07-01",
  "system_prompt_version": "v2.3",
  "temperature": 0.3,
  "latency_ms": 340,
  "human_reviewed": false,
  "feedback": null
}
```

---

## Lab: Usability Test and Iteration

### Goal
Run a usability test on your deployed model's interface, identify 3 issues, and fix them.

### Steps
1. **Define test tasks** (3-5 tasks users should be able to do)
2. **Run with 3-5 people** (friends, colleagues — observe silently)
3. **Document issues** (what confused them, what broke)
4. **Prioritize fixes** (severity + frequency matrix)
5. **Iterate** (fix top 2 issues, re-test)

### Deliverable
`ux_report.md` with test plan, findings, and before/after comparison.

---

## Exercises

1. **Confidence UI**  
   Sketch 3 different ways to show a model's confidence in its output to a user.

2. **Error Recovery**  
   The model generates a wrong answer. Design the UI flow for the user to correct it.

---

## Assignment (Graded)

### Task
Design and test the UX for a GenAI feature, incorporating human-in-the-loop controls.

### Requirements
- Wireframes or mockups for the interface
- HITL decision logic (when to auto-approve vs send to human)
- Audit trail schema
- Usability test results with at least 3 findings

### Deliverable
- `ux_report.md`

### Rubric (100 points)
- **UX design (30 points)**: Clear, trustworthy, user-friendly
- **HITL logic (25 points)**: Appropriate thresholds, sensible fallbacks
- **Audit trail (20 points)**: Complete, privacy-aware
- **Usability evidence (25 points)**: Real findings, prioritized fixes

---

# Capstone: Train and Ship a Language Model From Scratch

This is the final deliverable of the course. The full technical spec is in `capstone.md`. This section defines the rules, the phases, the gates, the rubric, and the defense.

The capstone combines everything from all 6 months:
- **Architecture** (Month 1): Decoder-only transformer design
- **Engineering** (Month 2): Clean code, modular design, tests
- **Training** (Month 3): Training loop, hyperparameter tuning
- **Data** (Month 4): Dataset curation for a domain
- **Evaluation** (Month 5): Perplexity, generation benchmarks, baseline comparisons
- **Deployment** (Month 6): API server, Docker, monitoring, UX

## 1. Rules

A violation of any rule is an automatic capstone failure (grade 0), regardless of everything else you built.

| # | Rule |
|---|------|
| R1 | The model must be implemented in pure PyTorch. No HF `Trainer`, no `transformers` model classes, no `keras`, no `timm`. |
| R2 | Training must start from random initialization. Loading any pretrained weights (yours or others') is not allowed. |
| R3 | The tokenizer must be a byte-level BPE you implement yourself. `tokenizers` and `tiktoken` are not allowed for training or encoding. You may use only `regex` for pre-tokenization. |
| R4 | Every component (RoPE, GQA, KV cache, sampling, training loop, serving) must have unit tests you wrote. |
| R5 | The full project must run from a fresh clone on the course machine with the provided Makefile (Gate G1). |
| R6 | Your git history must show incremental development. A single giant "everything" commit will be investigated. |
| R7 | You must not depend on a GPU to run tests or the demo. Tests and serving must work on CPU. |
| R8 | All code, docs, and reports must be in the repo and referenced from the README. |

## 2. The Deliverables (What the Repo Must Contain)

Every item below is required. Missing items are deductions; missing **core** items (marked ⚠) are automatic capstone failure.

```
capstone/
├── README.md                     # ⚠ Architecture diagram, design decisions, results, cost analysis, failure log
├── Makefile                      # ⚠ `make all`, `make test`, `make serve`, `make reproduce`, `make lint`, `make loadtest`
├── requirements.txt              # Pinned, with exact versions (hashes recommended)
├── pyproject.toml                # Project metadata + ruff, mypy, pytest config
├── .github/workflows/ci.yml      # CI pipeline (lint -> type -> test -> docker build -> eval gate)
├── config.py                     # All hyperparameters in one place (dataclass)
├── data/
│   ├── download.py               # ⚠ Reproducible data download
│   ├── clean.py                  # ⚠ Dedup + quality filter (documented thresholds)
│   ├── train.bin / val.bin / test.bin
│   └── dataset_card.md           # Size, source, licenses, quality stats
├── tokenizer/
│   ├── bpe.py                    # ⚠ Your BPE: pre-tokenize, train, encode, decode
│   ├── files/                    # Trained vocab + merge rules (committed)
│   └── test_bpe.py               # ⚠ Round-trip, determinism, special-token tests
├── model/
│   ├── rmsnorm.py
│   ├── rope.py                   # ⚠ precompute + apply_rope
│   ├── attention.py              # ⚠ GQA + KV cache (train & inference paths)
│   ├── swiglu.py
│   ├── block.py
│   ├── gpt.py                    # ⚠ Full model + generate() with temp/top-k/top-p
│   └── test_model.py             # ⚠ RoPE correctness, cache equivalence, masking, gradcheck
├── train/
│   ├── train.py                  # ⚠ Full training loop (see Phase 3 requirements)
│   ├── data.py                   # Streaming token-chunk reader (no loading everything in RAM)
│   ├── checkpoints/              # latest.pt + rolling history (keep at least 3)
│   └── logs/                     # Training curves (CSV or tensorboard events)
├── eval/
│   ├── perplexity.py
│   ├── benchmark.py              # 20-prompt generation benchmark
│   ├── judge.py                  # LLM-as-judge + human rating harness
│   └── eval_report.md            # ⚠ Results vs all baselines (see Phase 4)
├── serve/
│   ├── api.py                    # ⚠ FastAPI app (see Phase 5)
│   ├── server.py                 # Model load + batching + KV-cache pool
│   ├── cache.py                  # Semantic cache
│   ├── streaming.py              # SSE streaming
│   ├── audit.py                  # Audit-trail writer
│   ├── metrics.py                # Prometheus metrics
│   ├── ui.py                     # ⚠ Gradio/Streamlit UI
│   └── test_api.py               # ⚠ API tests (generate, stream, health, validation)
├── ops/
│   ├── Dockerfile                # ⚠ Multi-stage, non-root, pinned base
│   ├── docker-compose.yml        # api + ui + prometheus + grafana
│   ├── prometheus.yml
│   ├── grafana/dashboard.json
│   ├── locustfile.py             # Load-test script
│   └── incident_playbook.md
└── samples/                      # Generated outputs: before_training.txt, after_training.txt
```

## 3. Phase 0 — Proposal (Day 1, 10 points)

Before writing code, you submit a proposal that must be **approved** before you can continue. A rejected proposal costs you a day; you cannot skip it.

Your proposal must contain:

1. **Domain choice** (one of: code, medical, legal, finance, or your own niche). Justify it with a real product or job-market argument, not "it sounds cool."
2. **Data plan**: exact source, expected size in GB and tokens, license, download method, dedup/filtering strategy. 200 MB–2 GB of text.
3. **Architecture spec**: `vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `max_seq_len`, total params (10–50M), and total token budget. Every number must be justified against your compute budget.
4. **Compute budget**: where you train (Colab T4, own GPU, rented), how many hours you can actually commit, and how many tokens that implies.
5. **Risk register**: the 5 things most likely to fail (OOM, slow data pipeline, NaN loss, RoPE bug, scheduler bug) and your mitigation for each.
6. **Weekly milestones** mapping to the nine gates in Section 11.

> Expect questions like "Why 6 layers and not 8? Prove it from your token and time budget." Numbers must be defensible.

## 4. Phase 1 — The Tokenizer (Week 1, 25 points)

Build a byte-level BPE tokenizer **from scratch**. You may use only the Python standard library plus `regex` (for the GPT-2 pre-tokenization pattern). Everything else — byte fallback, pair counting, the merge loop, ranked encode, merge-lookup decode — is yours.

### Requirements (all tested, Gate G2)

1. **Pre-tokenization** using the GPT-2 regex pattern. Your splits must match reference behavior on a provided torture test (unicode, emoji, code, contractions, multi-byte text).
2. **Byte-level base**: the initial vocab is bytes 0–255, so *any* UTF-8 input round-trips.
3. **Merge training** to a configurable `vocab_size`, with a deterministic tie-break (documented) and a saved `merges` list.
4. **Special tokens**: `<|endoftext|>` (and any others you need) must never be split and must encode/decode correctly.
5. **Round-trip guarantee**: for a held-out corpus, `decode(encode(text)) == text` for 100% of documents. Prove it in a test.
6. **Determinism**: same corpus + seed ⇒ byte-identical vocab and merges.
7. **Efficiency**: encoding must tokenize your full corpus in under ~30 minutes. Profile it and report tok/s.

### What will be checked

- The merge loop is *your* algorithm. A reviewer will ask why it is O(P) rather than O(P log P), and how you find the most frequent pair across all sequences without re-scanning the entire corpus every iteration.
- The vocab/merges files are actually produced by your `train()` — no `tokenizers` artifacts hidden in the repo.
- Edge cases: empty string, whitespace-only, lone surrogates, extremely long repeated runs.

### Deliverables

`tokenizer/bpe.py`, `tokenizer/files/`, `tokenizer/test_bpe.py`, and a README section explaining your data structures and complexity.

## 5. Phase 2 — The Model (Week 1–2, 30 points)

Implement the full GPT-style decoder-only transformer: **RMSNorm, RoPE, Grouped Query Attention with a KV cache, SwiGLU, residual stream, weight tying, and an autoregressive `generate()`**. You must implement the naive attention path yourself for training; `F.scaled_dot_product_attention` is allowed only if you also pass a flash/no-flash equivalence test.

### Correctness requirements (all unit-tested, Gate G2)

| Test | What it proves |
|---|---|
| **RoPE rotation** | `apply_rope` on a pair of positions numerically equals a hand-written rotation-matrix reference. |
| **Position sensitivity** | Attention scores between identical-token pairs at different positions must differ (proves position information flows). |
| **Cache equivalence** | Generation with KV cache produces *byte-identical* tokens to generation without cache, for sequences of length ≥ 64. Non-negotiable. |
| **GQA shapes** | Q heads = `n_heads`, K/V heads = `n_kv_heads`, grouped broadcast correct (`n_heads % n_kv_heads == 0`). |
| **Weight tying** | `lm_head.weight is token_embedding.weight` (same tensor object). |
| **Masking** | With a causal mask, logits at position `t` are unchanged when token `t+1` is altered. |
| **Gradient check** | `torch.autograd.gradcheck` passes on a tiny model for RMSNorm, attention, SwiGLU, and the block. |
| **Sampling** | temperature=0 returns argmax; top-k=1 returns argmax; top-p=1.0 is a no-op; seeded generations are reproducible. |

### The KV cache

- Train time: no cache (full-context forward).
- Inference time: the cache must be **growable** (not pre-allocated to `max_seq_len` and ignored), correct under `use_cache=True`, and support **prefix reuse** (cache the prompt, then generate from it).
- `generate()` must accept `temperature`, `top_k`, `top_p`, `seed`, `max_new_tokens`, and stream token-by-token via a callback (required by the streaming server later).

### Performance requirement

On the course-standard machine (Gate G4), generation must meet the latency/throughput gates. Use an efficient forward path; per-token Python loops with repeated tensor allocations will not meet them.

### Deliverables

`model/*.py`, `model/test_model.py`, and README sections on architecture choices (RoPE vs learned, GQA, SwiGLU, weight tying) — *your* analysis, not copy-paste.

## 6. Phase 3 — Data Pipeline + Training (Week 2–3, 25 points)

### 6.1 Data pipeline

- `data/download.py` must be reproducible (pinned URLs/versions, license notes).
- `data/clean.py` must implement at least: exact dedup, a quality heuristic filter (avg token length, punctuation/emoji ratios, newline ratio), and a train/val/test split **with the same distribution** (document how you prevent leakage).
- Tokenize once; save as `train.bin` / `val.bin` / `test.bin` (raw `uint16`/`uint32` token ids).
- **Memory rule**: `train/data.py` must stream chunks (`mmap` or sequential reads). The whole tokenized corpus must **not** be materialized in RAM.

### 6.2 Training loop — write every line yourself

Your loop must include, and prove with logs:

1. AdamW with **decoupled** weight decay (no decay on norms/biases/embeddings), default 0.1.
2. Cosine schedule with linear warmup — implement the schedule yourself; be able to derive its formula on a whiteboard.
3. Mixed precision (AMP) with gradient scaling, in the correct order: scale → backward → unscale → clip → step → update.
4. Gradient clipping to a norm you justify.
5. Gradient accumulation, so you can reach large effective batch sizes on a small GPU.
6. Checkpointing: `latest.pt` + rolling history; **resume** must restore step counter, optimizer state, and scheduler state exactly.
7. Deterministic data ordering with seeds everywhere.
8. NaN/Inf guard: detect, log, and recover without silently corrupting the run.
9. Logging: step, loss, lr, grad norm, tokens/sec, GPU memory — to CSV or tensorboard. Every 1,000 steps, generate a sample and commit it under `samples/`.

### 6.3 The quality gate (Gate G3)

The course provides a **hidden reference checkpoint**: a model trained with the same budget (identical token budget, similar hyperparameter space) on the same processed dataset, published only as a perplexity number. Your submitted best checkpoint must either:

- Beat the reference val perplexity by **≥ 10% relative**, **or**
- Tie within **3%** *if* your parameter count is ≤ 80% of the reference's (you win on efficiency, not on size).

If your tokenizer, model, or loop has a silent bug, you land *above* the reference and fail. The reference is published only in the last week, so you cannot reverse-engineer it.

> In practice the gap to the reference is decided by data quality. The 10% usually comes from dedup and filtering, not from architecture.

### Deliverables

`train/train.py`, `train/data.py`, `data/clean.py`, checkpoints, training logs/curves, and a training report section in `eval_report.md` (loss curve, tokens seen, wall-clock time, what broke and how you fixed it).

## 7. Phase 4 — Evaluation (Week 3, 20 points)

### 7.1 Perplexity

`eval/perplexity.py` on the **test split** (untouched by training). Report loss and perplexity with token counts.

### 7.2 Generation benchmark

Create 20 diverse, *domain-specific* prompts (at least 3 "trap" prompts requiring format compliance, e.g., valid JSON or compilable code). For each:

- Generate with a **fixed seed** and temperature 0.7 (reproducible).
- Score 1–5 on **factual accuracy**, **coherence**, and **format compliance** — by you, by 2 independent raters, and by an LLM judge (`eval/judge.py`). Document inter-rater agreement (Cohen's κ) honestly.

### 7.3 Baseline table

| Metric | Your model | Reference model | Fine-tuned LLM (Month 3) | GPT-4o-mini (API) |
|---|---|---|---|---|
| Test perplexity | | | N/A | N/A |
| Latency (100 tokens) | | | | |
| Throughput (tok/s) | | | | |
| Size on disk | | | | |
| Training cost | | | | |
| Gen quality (1–5) | | | | |

Include a paragraph of **honest tradeoff analysis**: where your model wins, where it loses, and when you would choose each option in production.

### Deliverables

`eval/*.py`, `eval/eval_report.md`, `samples/after_training.txt`, and the baseline table.

## 8. Phase 5 — Serving (Week 3–4, 30 points)

### 8.1 FastAPI server (`serve/api.py`)

Required endpoints — all tested, all working on **CPU**:

| Endpoint | Behavior |
|---|---|
| `POST /generate` | JSON in / JSON out, validated parameters (bounds on `temperature`, `max_tokens`, `top_k`), returns output, token counts, latency, and a `request_id`. |
| `POST /stream` | **SSE streaming** of tokens as produced, reusing the prompt KV cache. |
| `GET /health` | Model + tokenizer loaded, cache warm. Returns version and uptime. |
| `GET /metrics` | Prometheus format (Phase 6). |

### 8.2 Server internals

1. **KV-cache pool**: generating from an extended prompt must reuse the cached prompt prefix. Measure and report the speedup.
2. **Semantic cache**: identical or near-identical prompts (similarity ≥ 0.95) must hit a cache and skip the model. Measure hit rate and savings.
3. **Batching**: implement request batching so concurrent `/generate` calls share a forward pass. Static batching of queued requests is the minimum; **continuous batching** (fill freed slots) is required for distinction.
4. **Backpressure**: bounded queue; when saturated, return `429` with `Retry-After` instead of unbounded growth.
5. **Streaming correctness**: time-to-first-token must be visibly lower than buffered mode. Prove it with a measurement.

### 8.3 Load test (Gate G5)

Write `ops/locustfile.py` and run a **30-minute soak test** with 50 concurrent users mixing `/generate` and `/stream`. Commit results showing:

- p50/p95/p99 time-to-first-token and time-to-last-token
- 0 HTTP 5xx errors, 0 dropped requests
- **Stable memory** (flat RSS curve; a leak is a failure)
- Throughput in requests/s

### 8.4 UI (`serve/ui.py`)

Gradio or Streamlit app that:
- Streams output token-by-token.
- Exposes temperature / top-k / top-p controls.
- Shows generation latency and token counts.
- Supports **edit-and-resubmit** (edit the model's output and re-ask).
- Lets the user **rate** outputs (thumbs up/down), persisted to the audit store.

### Deliverables

`serve/*.py`, `ops/locustfile.py`, load-test results, `ops/Dockerfile`, `ops/docker-compose.yml`.

## 9. Phase 6 — Observability, Safety, UX (Week 4, 30 points)

### 9.1 Observability

- **Structured logs**: JSON, one line per request, with `request_id`, `user_id` (if any), model version, input/output token counts, latency breakdown (preprocess / inference / postprocess), cache hit/miss, errors. No plain `print()`.
- **Metrics** (Prometheus): request count, latency histogram, error rate, tokens/sec, cache hit rate, queue depth, memory, estimated cost per request.
- **Tracing**: every request carries a trace/span chain through `api → batch → model` (OpenTelemetry or manual IDs).
- **Grafana dashboard** (`ops/grafana/dashboard.json`): latency percentiles, error rate, throughput, cache hit rate, token usage, cost. Screenshot committed to the repo.
- **Alert rules** (Prometheus rules file) + `ops/incident_playbook.md` covering at least 3 scenarios (latency spike, error-rate spike, memory leak).

### 9.2 Audit trail

Every interaction (input, output, params, model version, timing, rating) is written to an append-only store (`serve/audit.py`) with a documented schema. It must be **queryable** — a reviewer must be able to ask "show me every request from user X on this day with their ratings" and get an answer. Include a small query CLI or script.

### 9.3 Safety & guardrails

- Input validation (length caps, schema) and **rate limiting** per user/IP (configurable).
- **Output filter**: blocklist + hard max-token cap.
- Timeouts at every layer; sane error codes; no stack traces leaked to clients.
- A "known weaknesses" section in the README: what your model does badly and how you'd mitigate it in production.

### 9.4 HITL + UX

- Wireframes/mockups for the UI (committed).
- **HITL decision logic**: define a confidence signal (e.g., mean token log-prob, or a small auxiliary classifier) and a threshold below which outputs are held for human review instead of returned. Implement it server-side; the review queue can be a simple JSON store with a reviewer view.
- **Usability test**: 3–5 real users, 3–5 tasks, observe silently, document ≥ 3 issues, fix the top 2, re-test. `ux_report.md` with before/after.

### Deliverables

`serve/metrics.py`, `serve/audit.py`, `ops/prometheus.yml`, `ops/grafana/dashboard.json`, `ops/incident_playbook.md`, `ux_report.md`, dashboard screenshot, rate limiting + filters in `api.py`.

## 10. Phase 7 — Hardening & CI/CD (Week 4, 10 points)

- `pyproject.toml` with **ruff** (lint), **mypy/pyright** (type), and **pytest** configs — all clean, zero warnings.
- **CI** in `.github/workflows/ci.yml`: on every push — lint → typecheck → unit tests → docker build → **eval gate** (re-run test-split perplexity; fail CI on > 2% regression). CI must pass on your final commit.
- **Coverage ≥ 80%** on `tokenizer`, `model`, `eval`, `serve` (measured and reported).
- `Makefile`: `make all` runs the whole pipeline (data → tokenizer → train → eval → serve); `make reproduce` rebuilds your best checkpoint from raw data.
- `README.md` must contain: architecture diagram (ASCII or image), every design decision with reasoning, the cost/latency table, the failure log (what broke, what you learned), and clear run instructions.

## 11. The Gates (Hard Pass/Fail)

Each gate is binary. **Any red gate = capstone not passed** until it goes green. No partial credit on a gate.

| # | Gate | How it's measured | Pass condition |
|---|---|---|---|
| G1 | Reproducibility | Fresh clone on the course machine, `make reproduce` | Reproduces your submitted best within 0.05 val loss. |
| G2 | Correctness | `pytest` (tokenizer, model, api) | All green, including RoPE, cache-equivalence, masking, gradcheck. |
| G3 | Quality | Val perplexity vs hidden reference | Beat reference by ≥ 10% relative, or tie within 3% at ≤ 80% of its params. |
| G4 | Speed | Course-standard machine, cold start | TTFT < 250 ms, ≥ 15 tok/s at batch 1 (GPU or documented CPU profile). |
| G5 | Stability | 30-min locust soak, 50 users | p95 < 2 s, 0 errors, flat memory, no dropped requests. |
| G6 | Tests & CI | Coverage report + CI log | Coverage ≥ 80%; CI green on submitted commit. |
| G7 | Auditability | Manual inspection | Every request logged with trace id; audit store queryable. |
| G8 | History | `git log` | Incremental commits across ≥ 3 weeks; no one-shot repo. |
| G9 | Defense | 45-minute oral (Section 14) | Panel passes you. |

> G3 and G5 are the most commonly failed gates. Plan for them explicitly in Phase 0.

## 12. Stretch Goals (Distinction, up to +20 bonus)

Each item is verified in the defense. Points stack, but only if G1–G9 are all green.

- **Continuous batching** (+5): freed slots filled by queued requests mid-generation; proven with a concurrency plot.
- **Speculative decoding / draft model** (+4): measurable speedup at identical output quality.
- **Flash-attention / `scaled_dot_product_attention`** with flash/no-flash equivalence test (+3).
- **INT8 quantization** with < 1% perplexity regression and measured speedup (+3).
- **Distributed training** (DDP on ≥ 2 GPUs, or FSDP) with a correct scaling curve (+4).
- **Alignment-lite**: DPO or PPO-style preference tuning with measured win-rate vs base (+4).
- **Custom CUDA kernel** (e.g., fused RMSNorm+RoPE, or a KV-cache append kernel) (+5).
- **Automatic model routing** between your model and an API based on prompt complexity (+3).
- **Production extras**: rate-limited auth, multi-tenant namespacing, structured error taxonomy (+2 each, max +4).

Distinction requires ≥ 10 bonus points **and** Gate G3 beaten by ≥ 15% relative.

## 13. Grading Rubric (200 points)

| Category | Points | Notes |
|---|---|---|
| Phase 0 proposal | 10 | Depth, defensibility, risk awareness |
| Tokenizer | 25 | Correctness 15, determinism/efficiency 5, tests 5 |
| Model | 30 | Architecture 10, correctness tests 15, generation 5 |
| Data + training | 25 | Pipeline 10, loop quality 10, Gate G3 5 |
| Evaluation | 20 | Rigor, honest baselines, inter-rater stats |
| Serving | 30 | API 8, streaming + cache 8, batching 8, UI 6 |
| Observability | 15 | Logs/metrics/tracing 8, dashboard/alerts 7 |
| UX + HITL + audit | 15 | 5 each |
| Hardening + CI + docs | 10 | Lint/type/CI green, README quality |
| Defense | 10 | Oral performance |
| Stretch bonus | +20 max | Section 12 |

Pass threshold: **≥ 140/200**, all G1–G9 green, and no R1–R8 violation.

## 14. The Defense (45 minutes, mandatory, in person or video)

The defense is where the work is authenticated. Format:

- **10 min** — you present: architecture, decisions, results table, live demo (a real model, not a mockup).
- **25 min** — panel asks about your code. Expected questions include:
  - "Walk us through your merge loop. What's the worst-case complexity and why?"
  - "Derive the RoPE rotation formula. Why does it extrapolate to longer sequences?"
  - "Your `apply_rope` reshapes `x` a specific way — why does that matter? Show the tensor shapes at every step."
  - "Where exactly is the KV cache stored? How is it freed? What happens if two requests share a prompt?"
  - "Why does your loss curve have that hump at step 4,000?"
  - "Your cache-equivalence test — why that tolerance? What would a bug look like numerically?"
  - "What would you change with 5x more compute? With 5x less data? Defend it."
  - "Which would you ship for your domain — your model or GPT-4o-mini — and why?"
- **10 min** — panel gives feedback and verdict.

You may not bring notes written by anyone else. If you cannot explain a line in your repo, it is treated as not yours.

## 15. Timeline (4 weeks, day-by-day)

| Day | Work | Gate |
|---|---|---|
| 1 | Proposal draft + in-class review | Phase 0 |
| 2 | Proposal approved; data download started | Phase 0 |
| 3–4 | Pre-tokenizer + byte vocab + merge loop | Phase 1 |
| 5 | Encode/decode + special tokens; round-trip tests | Phase 1 |
| 6 | Tokenizer perf pass; vocab committed | Phase 1 |
| 7 | RMSNorm, RoPE + correctness tests | Phase 2 |
| 8 | GQA + causal mask + masking test | Phase 2 |
| 9 | SwiGLU, block, GPT, weight tying; gradcheck | Phase 2 |
| 10 | KV cache + cache-equivalence test | Phase 2 |
| 11 | `generate()` (temp/top-k/top-p/seed) + tests | Phase 2 |
| 12 | Data clean: dedup + filters + split + `.bin` | Phase 3 |
| 13 | Train loop v1 (AMP, clip, checkpoint, resume) | Phase 3 |
| 14 | Scheduler + grad accumulation + NaN guard | Phase 3 |
| 15 | First short run; fix bugs | Phase 3 |
| 16 | Main training run (long); start eval harness | Phase 3/4 |
| 17 | Perplexity + 20-prompt benchmark + raters | Phase 4 |
| 18 | Baseline table vs fine-tuned + API; `eval_report.md` | Phase 4 |
| 19 | FastAPI: `/generate`, `/health`, `/metrics` | Phase 5 |
| 20 | SSE `/stream` + streaming tests | Phase 5 |
| 21 | KV-cache reuse + semantic cache | Phase 5 |
| 22 | Batching + backpressure (429) | Phase 5 |
| 23 | UI (streaming, controls, rating, edit) | Phase 5 |
| 24 | Locust load test; fix; re-run 30-min soak | G5 |
| 25 | Metrics + logs + tracing + Grafana + alerts | Phase 6 |
| 26 | Audit store + query tool + rate limiting + filters | Phase 6 |
| 27 | HITL logic + review queue + usability test | Phase 6 |
| 28 | Fix top-2 UX issues; re-test | Phase 6 |
| 29 | CI/CD, coverage, Makefile, README, samples | Phase 7 |
| 30 | `make reproduce` clean run + gate check | G1–G8 |
| 31 | Defense prep + mock defense with a peer | Defense |
| 32 | **Submission + defense** | G9 |

## 16. Penalties and Auto-Fails

- **Any R1–R8 violation**: grade 0. No exceptions.
- **Any red gate at submission**: not passed until fixed and re-graded (one re-grade allowed, with late penalty).
- **Late submission**: 5 points per day after a 2-day grace period.
- **Reports padded with fluff / LLM-generated filler without analysis**: up to half the report's points deducted.
- **Missing core deliverables (⚠ in Section 2)**: auto-fail of the relevant gate.

## 17. Submission Checklist

- [ ] `make all` and `make reproduce` work from a fresh clone (G1)
- [ ] `pytest` green, coverage ≥ 80% (G2, G6)
- [ ] Val perplexity beats reference by ≥ 10% (or efficiency tie) (G3)
- [ ] TTFT < 250 ms, ≥ 15 tok/s (G4)
- [ ] 30-min soak: p95 < 2 s, 0 errors, flat memory (G5)
- [ ] CI green on final commit (G6)
- [ ] Audit store queryable; every request traced (G7)
- [ ] Incremental git history (G8)
- [ ] README with architecture, decisions, tables, failure log
- [ ] `eval_report.md`, `ux_report.md`, `incident_playbook.md`, `optimization_report.md`, `slo_report.md`
- [ ] Defense booked and passed (G9)

## Assessment: Quick Quiz (5 Questions)

1. **What is KV caching and why does it matter?**  
   KV caching stores the key and value tensors from previous generation steps so they don't need to be recomputed. It reduces attention complexity from O(T^2) to O(T) per step, dramatically speeding up autoregressive generation.

2. **What's the difference between p50, p95, and p99 latency?**  
   p50 is the median latency (half of requests are faster). p95 means 95% of requests are at or below this latency. p99 is the 99th percentile. p99 matters most for user experience because it captures the tail — the slow requests that frustrate users.

3. **When should you require human review of model outputs?**  
   When stakes are high (medical, legal, financial decisions), when model confidence is low, for first N outputs of a new model version, or for edge cases outside the training distribution.

4. **What's the most impactful optimization for inference latency?**  
   For most systems, KV caching gives the biggest single improvement (up to 10x for long generations). Combined with FP16 inference, it typically covers 80% of the gains. Quantization helps further but with more engineering effort.

5. **What belongs in every audit log entry?**  
   Timestamp, user ID, input, output, model/version info, system prompt version, inference parameters, latency, whether human review occurred, and any user feedback.

---

## Common Pitfalls and How to Address Them

- **Optimizing before measuring**  
  Teams guess at bottlenecks instead of measuring. *Solution*: Always baseline first. Measure latency, memory, throughput before changing anything.

- **Ignoring tail latency**  
  Average latency looks great while p99 is terrible. *Solution*: Monitor p95 and p99, not just averages.

- **One-size-fits-all model selection**  
  Using GPT-4o for every request wastes money on simple tasks. *Solution*: Implement model routing — small models for simple queries, large models for complex ones.

- **No audit trail**  
  When something goes wrong, you can't trace it. *Solution*: Log every request with enough context to debug and audit.

- **Shipping and forgetting**  
  Models degrade over time (data drift, user behavior change). *Solution*: Monitor quality metrics continuously, schedule regular evaluations.

---

## Resources

- **Inference optimization**: vLLM docs, TensorRT-LLM, HuggingFace Optimum
- **Monitoring**: Prometheus + Grafana, LangFuse, Weights & Biases Prompts
- **UX research**: Nielsen Norman Group guidelines for AI interactions
- **Reliability**: Google SRE books, incident response playbooks
- **Deployment**: FastAPI docs, Docker best practices, GitHub Actions for CI/CD
