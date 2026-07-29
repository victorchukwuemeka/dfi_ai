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

See `capstone.md` for the full specification.

The capstone is the final deliverable of the course. It combines everything from all 6 months:
- **Architecture** (Month 1): Decoder-only transformer design
- **Engineering** (Month 2): Clean code, modular design
- **Training** (Month 3): Training loop, hyperparameter tuning
- **Data** (Month 4): Dataset curation for a domain
- **Evaluation** (Month 5): Perplexity, generation benchmarks, baseline comparisons
- **Deployment** (Month 6): API server, Docker, monitoring, UX

---

## Key Deliverables

| Week | Focus | Output |
|------|-------|--------|
| 1 | Tokenizer + dataset | `tokenizer.py`, processed data |
| 2 | Model + training loop | `model.py`, `train.py` running |
| 3 | Full training + eval | Checkpoints, `eval_report.md` |
| 4 | Serve + docs + deploy | `api.py`, `app.py`, Dockerfile, README |

---

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
