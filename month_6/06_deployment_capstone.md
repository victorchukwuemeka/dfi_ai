# Deployment and Capstone — Full Course Module

## Module Overview
This module takes learners from working prototypes to production-grade GenAI systems. The focus is on performance optimization, reliability engineering, observability, and human-centered design. The capstone ties everything together by building a production-grade GenAI product and shipping it.

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
| Capstone | Build + ship a production GenAI product | End-to-end project |

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

# Capstone: Build and Ship a Production-Grade GenAI Product

This is the final deliverable of the course. The full technical spec is in `capstone.md`. This section defines the rules, the phases, the gates, the rubric, and the defense.

The capstone combines everything from all 6 months:
- **Product + prompting** (Month 1): prompt system, output schemas, guardrails, inference config
- **RAG + agents** (Month 2): grounding, retrieval, reranking, function calling
- **Fine-tuning** (Month 3): dataset design and LoRA/QLoRA adaptation
- **Multimodal** (Month 4, optional): vision/audio inputs
- **Evaluation** (Month 5): LLM-as-judge, A/B testing, red-teaming
- **Deployment** (Month 6): API server, monitoring, UX

## 1. Rules

A violation of any rule is an automatic capstone failure (grade 0), regardless of everything else you built.

| # | Rule |
|---|------|
| R1 | The model must come from the taught stack: `transformers` + `peft` (open-weight) or `openai` (API). No home-grown from-scratch model is required, and no pre-built chain framework may replace your own application logic. |
| R2 | Adaptation must be parameter-efficient fine-tuning (LoRA/QLoRA) of a pretrained model. No training from random initialization; no full-model fine-tuning of large models. |
| R3 | Your code must be yours: RAG, agents, eval harness, API, cache, batching, and audit logic written by you. You may use the taught libraries (`openai`, `sentence-transformers`, `chromadb`, `peft`, `transformers`, `fastapi`) but not a framework that hides those components. |
| R4 | Every component (prompts, RAG, tools/agents, LoRA script, judge, API, cache, batching, streaming, audit) must have unit tests you wrote. |
| R5 | The full project must run from a fresh clone on the course machine with the provided Makefile (Gate G1). |
| R6 | Your git history must show incremental development. A single giant "everything" commit will be investigated. |
| R7 | You must not depend on a GPU to run tests or the demo. Tests and serving must work on CPU with a small open-weight model, or against the API model. |
| R8 | All code, docs, and reports must be in the repo and referenced from the README. |

## 2. The Deliverables (What the Repo Must Contain)

Every item below is required. Missing items are deductions; missing **core** items (marked ⚠) are automatic capstone failure.

```
capstone/
├── README.md                     # ⚠ Architecture diagram, design decisions, results, cost analysis, failure log
├── Makefile                      # ⚠ `make all`, `make test`, `make serve`, `make reproduce`, `make lint`, `make loadtest`
├── requirements.txt              # Pinned, with exact versions (hashes recommended)
├── pyproject.toml                # Project metadata + pytest config
├── config.py                     # All hyperparameters and settings in one place (dataclass)
├── data/
│   ├── download.py               # ⚠ Reproducible data download
│   ├── clean.py                  # ⚠ Dedup + quality filter (documented thresholds)
│   ├── corpus/                   # Domain corpus for RAG + fine-tuning
│   └── dataset_card.md           # Size, source, licenses, quality stats
├── app/
│   ├── prompts.py                # ⚠ Prompt library (versioned templates)
│   ├── schema.py                 # Output schemas + validation
│   ├── rag.py                    # ⚠ Chunking, embeddings, retrieval, reranking, citations
│   ├── agent.py                  # ⚠ ReAct-style loop with guardrails
│   ├── tools.py                  # Function-calling tools + error handling
│   └── test_app.py               # ⚠ Tests for prompts, RAG, agents
├── finetune/
│   ├── prep_data.py              # ⚠ Clean + format + tokenize the training data
│   ├── train_lora.py             # ⚠ LoRA/QLoRA script (peft + HF Trainer, as taught)
│   ├── checkpoints/              # Adapter checkpoints (latest + history)
│   └── logs/                     # Training curves (CSV or tensorboard events)
├── eval/
│   ├── judge.py                  # ⚠ LLM-as-judge harness
│   ├── benchmark.py              # 20-prompt domain benchmark + trap prompts
│   ├── ab_test.py                # A/B analysis (t-test, effect size)
│   ├── eval_report.md            # ⚠ Results vs baseline (see Phase 4)
│   └── red_team_report.md        # ⚠ Attack + mitigation log
├── serve/
│   ├── api.py                    # ⚠ FastAPI app (see Phase 5)
│   ├── server.py                 # Model load + batching + KV-cache pool
│   ├── cache.py                  # Semantic cache
│   ├── streaming.py              # Token streaming (StreamingResponse)
│   ├── audit.py                  # Audit-trail writer
│   ├── metrics.py                # Prometheus metrics
│   ├── ui.py                     # ⚠ Gradio/Streamlit UI
│   └── test_api.py               # ⚠ API tests (generate, stream, health, validation)
├── ops/
│   ├── prometheus.yml
│   ├── grafana/dashboard.json
│   ├── soak_test.py              # Concurrent soak script (plain Python)
│   └── incident_playbook.md
└── samples/                      # Generated outputs: before_improvements.txt, after_improvements.txt
```

## 3. Phase 0 — Proposal (Day 1, 10 points)

Before writing code, you submit a proposal that must be **approved** before you can continue. A rejected proposal costs you a day; you cannot skip it.

Your proposal must contain:

1. **Domain + product choice** (one of: code, medical, legal, finance, or your own niche). Define the concrete user task and justify it with a real product or job-market argument, not "it sounds cool."
2. **Data plan**: exact source, expected size, license, download method, dedup/filtering strategy, and how you will prevent train/val/test leakage.
3. **Model + adaptation plan**: which base model (API or open-weight) and why; whether you will need LoRA; the prompting/RAG/tools architecture.
4. **Compute budget**: where you run (CPU, Colab T4, own GPU, rented), how many hours you can commit, and what that implies for the LoRA run.
5. **Evaluation plan**: how you will measure quality (LLM-judge rubric, domain metrics) and what baseline you will beat.
6. **Risk register**: the 5 things most likely to fail (leaky data split, RAG retrieval gaps, LoRA overfitting, slow retrieval, load-test failures) and your mitigation for each.
7. **Weekly milestones** mapping to the nine gates in Section 11.

> Expect questions like "Why LoRA rank 8 and not 16?" and "How do you know your retrieval is good enough?" Numbers must be defensible.

## 4. Phase 1 — Product Core and Prompting Baseline (Week 1, 20 points)

Build the product definition and the first working system, using the Month 1 toolkit.

1. **Prompt library** (`app/prompts.py`): role, task, constraints, output schema, few-shot examples; versioned system prompts.
2. **Inference configuration**: document temperature / top-k / top-p per task type, justified from your Month 1 work.
3. **Baseline system**: the simplest prompt + model pipeline that does the task acceptably. Generate and commit samples.
4. **Guardrails**: input validation, length caps, schema enforcement with failover, retry loops, refusal rules.
5. **Baseline measured**: quality score (LLM-judge or domain metric) and latency recorded for the comparison table.

### What will be checked

- Every prompt template has a test that the filled prompt satisfies its documented contract.
- Schema enforcement is at the application level (parse + validate + failover), never a crash.
- The baseline is real and reproducible, not a screenshot.

### Deliverables

`app/prompts.py`, `app/schema.py`, `app/test_app.py`, `samples/before_improvements.txt`, and a README section on prompt/inference design.

## 5. Phase 2 — RAG, Tools, and Agents (Week 1–2, 25 points)

Ground the product and give it capabilities, using the Month 2 stack. At least one of RAG or tools/agents is mandatory; a strong product usually needs both.

**RAG:**
- Corpus curation, chunking strategy documented, Chroma store with metadata.
- Retrieval (top-k) + reranking (cross-encoder or hybrid vector + BM25).
- Citation grounding: answers must cite retrieved chunks, and citations are verified.
- Metrics reported: Hit Rate, MRR, Citation Precision.

**Tools/agents:**
- Function-calling tool schemas with routing and retry/backoff.
- ReAct-style loop (Thought → Action → Observation) with step limits, token budgets, and guardrails.
- Every tool call logged with input, output, latency, error status.

### What will be checked

- Retrieval tests: the expected chunk is in top-k for labeled queries.
- Reranking tests: top-1 recall after rerank >= before rerank on the eval set.
- Citation tests: every citation marker maps to a retrieved chunk.
- Agent tests: max-step enforcement; a "stuck" agent errors cleanly.
- Retrieval quality numbers appear in `eval_report.md` — no hand-waving.

### Deliverables

`app/rag.py`, `app/agent.py`, `app/tools.py`, `data/` (corpus + download script), `app/test_app.py`.

## 6. Phase 3 — Dataset and LoRA Fine-Tuning (Week 2–3, 25 points)

A domain dataset and a LoRA fine-tune of a small open-weight model (Phi-2, Gemma-2B, TinyLlama), following the Month 3 playbook.

### 6.1 Dataset
- `data/download.py` reproducible (pinned URLs/versions, license notes).
- `data/clean.py` with exact dedup, documented quality filters, and a train/val/test split that prevents leakage (hash check in a test).
- `finetune/prep_data.py` formats examples to the model's chat template and tokenizes once; verifies token counts/lengths.

### 6.2 LoRA fine-tuning
- `LoraConfig` with documented `r`, `alpha`, `target_modules`, `dropout`, `task_type` — every number justified.
- HF `Trainer` (as taught) with gradient accumulation, `fp16`/`bf16`, checkpoints; W&B or tensorboard curves committed.
- Save the adapter (`save_pretrained`); never silently merge into the base.
- **Regression testing**: tuned adapter must not regress general capability while improving the domain task, on a fixed eval set.

### What will be checked
- Split integrity test passes (no document in two splits).
- `PeftModel.from_pretrained` reproduces your reported eval results.
- Loss curves + samples are in the repo; you can explain any spike.

### Deliverables

`data/*`, `finetune/*` (prep_data.py, train_lora.py, checkpoints, logs), and a training section in `eval_report.md`.

## 7. Phase 4 — Evaluation (Week 3, 25 points)

### 7.1 LLM-as-judge
`eval/judge.py` scores outputs 1–5 on the dimensions you define (factual accuracy, coherence, format compliance, domain usefulness). Control for the biases Month 5 teaches (position, verbosity, self-enhancement). Document the rubric.

### 7.2 Human rating
You plus 2 independent raters score a sample with the same rubric; report inter-rater agreement (simple agreement rate, as taught in Month 5) honestly.

### 7.3 Domain benchmark
20 diverse, domain-specific prompts, at least 3 "trap" prompts (valid JSON, format compliance). Fixed seed, fixed temperature. Outputs committed to `samples/`.

### 7.4 A/B comparison
Final system vs Part 1 baseline (and, if applicable, vs the untuned base model), with statistical analysis (t-test, effect size) as taught.

### 7.5 Red-team
Attack your own system (prompt injection, jailbreaks, data exfiltration); document findings and fixes in `eval/red_team_report.md`.

### Baseline table

| Metric | Baseline (Phase 1) | Base model (untuned) | Fine-tuned (Phase 3) | Final system (full stack) |
|---|---|---|---|---|
| LLM-judge score (1–5) | | | | |
| Domain metric (MRR / hit rate / accuracy) | | | | |
| Latency (100 tokens) | | | | |
| Throughput (tok/s) | | | | |
| Training cost | | | | |
| Error rate | | | | |

Include a paragraph of **honest tradeoff analysis**: where each configuration wins and when you would ship each in production.

### Deliverables

`eval/*.py`, `eval/eval_report.md`, `eval/red_team_report.md`, `samples/after_improvements.txt`, and the baseline table.

## 8. Phase 5 — Serving (Week 3–4, 30 points)

### 8.1 FastAPI server (`serve/api.py`)

Required endpoints — all tested, all working on **CPU** (small open-weight model) or against the API model:

| Endpoint | Behavior |
|---|---|
| `POST /generate` | JSON in / JSON out, validated parameters (bounds on `temperature`, `max_tokens`, `top_k`), returns output, token counts, latency, and a `request_id`. |
| `POST /stream` | **Token streaming** of tokens as produced (FastAPI `StreamingResponse`). |
| `GET /health` | Model + tokenizer loaded, cache warm. Returns version and uptime. |
| `GET /metrics` | Prometheus format (Phase 6). |

### 8.2 Server internals

1. **KV-cache pool**: generating from an extended prompt must reuse the cached prompt prefix. Measure and report the speedup.
2. **Semantic cache**: identical or near-identical prompts (similarity ≥ 0.95) must hit a cache and skip the model. Measure hit rate and savings.
3. **Batching**: implement request batching so concurrent `/generate` calls share a forward pass. Static batching of queued requests is the minimum; **continuous batching** (fill freed slots) is required for distinction.
4. **Backpressure**: bounded queue; when saturated, return `429` with `Retry-After` instead of unbounded growth.
5. **Streaming correctness**: time-to-first-token must be visibly lower than buffered mode. Prove it with a measurement.

### 8.3 Load test (Gate G5)

Write `ops/soak_test.py` — a plain-Python concurrent script (thread pool + `httpx`/`requests`, no extra framework) — and run a **30-minute soak test** with 50 concurrent clients mixing `/generate` and `/stream`. Commit results showing:

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

`serve/*.py`, `ops/soak_test.py`, soak-test results.

## 9. Phase 6 — Observability, Safety, UX (Week 4, 30 points)

### 9.1 Observability

- **Structured logs**: JSON, one line per request, with `request_id`, `user_id` (if any), model version, input/output token counts, latency breakdown (preprocess / inference / postprocess), cache hit/miss, errors. No plain `print()`.
- **Metrics** (Prometheus): request count, latency histogram, error rate, tokens/sec, cache hit rate, queue depth, memory, estimated cost per request.
- **Tracing**: every request carries a trace/span chain through `api → pipeline → model` (OpenTelemetry or manual IDs).
- **Grafana dashboard** (`ops/grafana/dashboard.json`): latency percentiles, error rate, throughput, cache hit rate, token usage, cost. Screenshot committed to the repo.
- **Alert rules** (Prometheus rules file) + `ops/incident_playbook.md` covering at least 3 scenarios (latency spike, error-rate spike, memory leak).

### 9.2 Audit trail

Every interaction (input, output, params, model version, timing, rating) is written to an append-only store (`serve/audit.py`) with a documented schema. It must be **queryable** — a reviewer must be able to ask "show me every request from user X on this day with their ratings" and get an answer. Include a small query CLI or script.

### 9.3 Safety & guardrails

- Input validation (length caps, schema) and **rate limiting** per user/IP (configurable).
- **Output filter**: blocklist + hard max-token cap.
- Timeouts at every layer; sane error codes; no stack traces leaked to clients.
- A "known weaknesses" section in the README: what your system does badly and how you'd mitigate it in production.
- Red-team findings and fixes from `eval/red_team_report.md` reflected in the code.

### 9.4 HITL + UX

- Wireframes/mockups for the UI (committed).
- **HITL decision logic**: define a confidence signal (e.g., mean token log-prob, or a small auxiliary classifier) and a threshold below which outputs are held for human review instead of returned. Implement it server-side; the review queue can be a simple JSON store with a reviewer view.
- **Usability test**: 3–5 real users, 3–5 tasks, observe silently, document ≥ 3 issues, fix the top 2, re-test. `ux_report.md` with before/after.

### Deliverables

`serve/metrics.py`, `serve/audit.py`, `ops/prometheus.yml`, `ops/grafana/dashboard.json`, `ops/incident_playbook.md`, `ux_report.md`, dashboard screenshot, rate limiting + filters in `api.py`.

## 10. Phase 7 — Hardening & Final Docs (Week 4, 10 points)

- `make check`: one command that runs the full test suite plus the domain benchmark regression check (fail on > 2% regression vs the committed baseline, as in Month 3 regression testing). Must pass on your final commit.
- `Makefile`: `make all` runs the whole pipeline (data → finetune → eval → serve); `make reproduce` rebuilds the app and eval numbers from a fresh clone; `make soak` runs the soak test.
- `README.md` must contain: architecture diagram (ASCII or image), every design decision with reasoning, the cost/latency table, the failure log (what broke, what you learned), and clear run instructions.

## 11. The Gates (Hard Pass/Fail)

Each gate is binary. **Any red gate = capstone not passed** until it goes green. No partial credit on a gate.

| # | Gate | How it's measured | Pass condition |
|---|---|---|---|
| G1 | Reproducibility | Fresh clone on the course machine, `make reproduce` | App builds and reported eval numbers reproduce within tolerance. |
| G2 | Correctness | `pytest` (app, eval, serve) | All green, including prompts, RAG, agents, judge, API, cache, batching. |
| G3 | Quality | Domain benchmark vs documented baseline | ≥ 60% LLM-judge win rate (or an equivalent, pre-approved domain metric margin). |
| G4 | Speed | Course-standard machine | TTFT < 250 ms, ≥ 15 tok/s at batch 1 (API or documented local model). |
| G5 | Stability | 30-min soak, 50 concurrent clients | p95 < 2 s, 0 errors, flat memory, no dropped requests. |
| G6 | Tests | `pytest` + `make check` | All green on submitted commit. |
| G7 | Auditability | Manual inspection | Every request logged with trace id; audit store queryable. |
| G8 | History | `git log` | Incremental commits across ≥ 3 weeks; no one-shot repo. |
| G9 | Defense | 45-minute oral (Section 14) | Panel passes you. |

> G3 and G5 are the most commonly failed gates. Plan for them explicitly in Phase 0.

## 12. Stretch Goals (Distinction, up to +20 bonus)

Each item is verified in the defense. Points stack, but only if G1–G9 are all green.

- **Continuous batching** (+5): freed slots filled by queued requests mid-generation; proven with a concurrency plot.
- **Multimodal integration** (+4): vision/audio input in your pipeline (Month 4 stack).
- **Advanced RAG** (+3): query rewriting, HyDE, hybrid search, citation verification.
- **INT8/4-bit quantized inference** (+3): < 1% quality regression, measured speedup.
- **Multi-agent orchestration** (+4): planner + worker agents with verification.
- **In-production A/B testing** (+3): traffic split with statistical analysis.
- **Automatic model routing** (+3): small/cheap model vs large model by prompt complexity.
- **Custom domain eval metric** (+3): designed, validated, and regression-gated in `make check`.
- **Production extras**: rate-limited auth, multi-tenant namespacing, structured error taxonomy (+2 each, max +4).

Distinction requires ≥ 10 bonus points **and** Gate G3 beaten by a wider margin (≥ 75% win rate).

## 13. Grading Rubric (200 points)

| Category | Points | Notes |
|---|---|---|
| Phase 0 proposal | 10 | Depth, defensibility, risk awareness |
| Product core + prompting | 20 | Prompt system, schemas, guardrails, baseline |
| RAG + tools/agents | 25 | Retrieval quality, grounding, agent correctness |
| Data + fine-tuning | 25 | Pipeline, LoRA quality, regression testing |
| Evaluation | 25 | Judge harness, human raters, A/B, red-team |
| Serving | 30 | API 8, streaming + cache 8, batching 8, UI 6 |
| Observability | 15 | Logs/metrics/tracing 8, dashboard/alerts 7 |
| UX + HITL + audit | 20 | 5 each (UX, HITL, audit, safety) |
| Hardening + docs | 15 | Clean code, tests green, README quality |
| Defense | 10 | Oral performance |
| Stretch bonus | +20 max | Section 12 |

Pass threshold: **≥ 140/200**, all G1–G9 green, and no R1–R8 violation.

## 14. The Defense (45 minutes, mandatory, in person or video)

The defense is where the work is authenticated. Format:

- **10 min** — you present: architecture, decisions, results table, live demo (a real system, not a mockup).
- **25 min** — panel asks about your code. Expected questions include:
  - "Walk us through your RAG pipeline. Why this chunk size? Where does grounding break?"
  - "Explain LoRA. Why rank 8 and alpha 16? What does the delta matrix capture?"
  - "How does your LLM-as-judge work? What biases did you control for, and how?"
  - "Where is the semantic cache keyed? Why a 0.95 cosine threshold?"
  - "Where exactly is the KV cache stored? How is it freed? What happens if two requests share a prompt?"
  - "Which optimization gave the biggest latency win — cache, batching, or streaming? Prove it with your numbers."
  - "Show me your A/B analysis. What does the t-test actually tell you here?"
  - "What would you change with 5x more data? With 5x less compute? Defend it."
  - "Where does your system fail? What is in your known-weaknesses section?"
- **10 min** — panel gives feedback and verdict.

You may not bring notes written by anyone else. If you cannot explain a line in your repo, it is treated as not yours.

## 15. Timeline (4 weeks, day-by-day)

| Day | Work | Gate |
|---|---|---|
| 1 | Proposal draft + in-class review | Phase 0 |
| 2 | Proposal approved; data download started | Phase 0 |
| 3–4 | Product definition + prompt library + schema | Phase 1 |
| 5 | Inference config + baseline system + samples | Phase 1 |
| 6 | Guardrails + validation tests | Phase 1 |
| 7 | Corpus chunking + embedding + Chroma store | Phase 2 |
| 8 | Retrieval + reranking + retrieval tests | Phase 2 |
| 9 | Citations/grounding + retrieval metrics | Phase 2 |
| 10 | Tools/agents (schemas, loop, guardrails) | Phase 2 |
| 11 | Data cleaning + split + chat format | Phase 3 |
| 12 | LoRA training run #1 (short) | Phase 3 |
| 13 | Fix dataset/format bugs; rerun | Phase 3 |
| 14 | Full LoRA run + regression tests | Phase 3 |
| 15 | Judge harness + rubric + inter-rater agreement | Phase 4 |
| 16 | 20-prompt benchmark + trap prompts | Phase 4 |
| 17 | A/B vs baseline + statistical analysis | Phase 4 |
| 18 | Red-team pass + fixes; eval_report.md | Phase 4 |
| 19 | FastAPI: `/generate`, `/health`, `/metrics` | Phase 5 |
| 20 | Streaming `/stream` + streaming tests | Phase 5 |
| 21 | KV-cache reuse + semantic cache | Phase 5 |
| 22 | Batching + backpressure (429) | Phase 5 |
| 23 | UI (streaming, controls, rating, edit) | Phase 5 |
| 24 | Soak test; fix; re-run 30-min soak | G5 |
| 25 | Metrics + logs + tracing + Grafana + alerts | Phase 6 |
| 26 | Audit store + query tool + rate limiting + filters | Phase 6 |
| 27 | HITL logic + review queue + usability test | Phase 6 |
| 28 | Fix top-2 UX issues; re-test | Phase 6 |
| 29 | `make check`, Makefile, README, samples | Phase 7 |
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
- [ ] `pytest` green and `make check` passes (G2, G6)
- [ ] Domain benchmark beats baseline by ≥ 60% judge win rate (or approved metric) (G3)
- [ ] TTFT < 250 ms, ≥ 15 tok/s (G4)
- [ ] 30-min soak: p95 < 2 s, 0 errors, flat memory (G5)
- [ ] `make check` green on final commit (G6)
- [ ] Audit store queryable; every request traced (G7)
- [ ] Incremental git history (G8)
- [ ] README with architecture, decisions, tables, failure log
- [ ] `eval_report.md`, `ux_report.md`, `incident_playbook.md`, `optimization_report.md`, `slo_report.md`, `red_team_report.md`
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
- **Deployment**: FastAPI docs
