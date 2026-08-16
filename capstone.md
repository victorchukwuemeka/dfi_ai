# Capstone: Build and Ship a Production-Grade GenAI Product

The final project of the course. You will build a working GenAI product end-to-end: a domain dataset, a prompting system, a RAG/tool pipeline, a LoRA fine-tune of a pretrained model, an evaluation harness, and a production server. No starter code is provided.

The full contract (rules, gates, rubric, defense) is in `month_6/06_deployment_capstone.md`. This file is the technical specification.

## Deliverable

A production-grade GenAI product in a domain you choose. It must:

1. Use a **pretrained** base model — an API model (`gpt-4o-mini` / `gpt-4o`) or an open-weight small model (Phi-2, Gemma-2B, TinyLlama) — via the taught stack (`openai`, `transformers`, `peft`). No training from random initialization.
2. Be grounded and tool-capable: a RAG pipeline (embeddings, retrieval, reranking) and/or tools/agents built with `sentence-transformers`, `chromadb`, and function calling.
3. Be adapted to your domain with **LoRA/QLoRA** fine-tuning on a dataset you curate and clean.
4. Be evaluated rigorously: an LLM-as-judge harness, a domain benchmark, an A/B comparison against a documented baseline, and a red-team pass.
5. Be served through a FastAPI server with streaming, caching, batching, monitoring, and a web UI.
6. Prove it works: unit tests, CI, load tests, and a live defense.

All code in one GitHub repo with a README covering every design decision.

## Rules

A violation of any rule is an automatic capstone failure (grade 0).

| # | Rule |
|---|------|
| R1 | The model must come from the taught stack: `transformers` + `peft` (open-weight) or `openai` (API). No home-grown from-scratch model is required, and no pre-built chain framework may replace your own application logic. |
| R2 | Adaptation must be parameter-efficient fine-tuning (LoRA/QLoRA) of a pretrained model. No training from random initialization, no full-model fine-tuning of large models. |
| R3 | Your code must be yours: RAG, agents, eval harness, API, cache, batching, and audit logic written by you. You may use the taught libraries (`openai`, `sentence-transformers`, `chromadb`, `peft`, `transformers`, `fastapi`) but not a framework that hides those components. |
| R4 | Every component must have unit tests you wrote: prompts, RAG, tools/agents, LoRA script, judge, API, cache, batching, streaming, audit. |
| R5 | The project must run from a fresh clone on the course machine via the provided Makefile. |
| R6 | Git history must show incremental development across the project. A one-shot "everything" commit will be investigated. |
| R7 | Tests and the demo must run on CPU with a small open-weight model, or against the API model. No GPU is required. |
| R8 | Everything lives in the repo and is referenced from the README. |

## Tech Stack

| Layer | Choice | Constraint |
|---|---|---|
| Base model | API (`gpt-4o-mini`, `gpt-4o`) or open-weight (Phi-2, Gemma-2B, TinyLlama) | Pretrained only — no random init |
| Tokenizer | The base model's tokenizer | No custom tokenizer required |
| Adaptation | LoRA / QLoRA via `peft` | Parameter-efficient only |
| Grounding | RAG with `sentence-transformers` + `chromadb`, rerankers, function calling | Build the pipeline yourself |
| Evaluation | LLM-as-judge, domain metrics, A/B analysis, red-team | Judge harness is yours |
| Serving | FastAPI + Gradio/Streamlit + Docker | SSE streaming, caching, batching, metrics |
| Tracking | W&B or local tensorboard | Log training + eval; publish your curves |
| Quality | LLM-as-judge win-rate + domain benchmark vs a documented baseline | Beat your baseline by a defined margin |

---

## Part 1: Product Core and Prompting Baseline (Week 1)

### What to Build

The product definition and the first working system. Everything after this is measured against it.

1. **Product definition**: pick ONE domain (code, medical, legal, finance, or your niche) and a concrete user task with real stakes. Justify it with a product or job-market argument.
2. **Prompt system**: role, task, constraints, output schema, few-shot examples, and a system prompt versioned like code. Build a prompt library of reusable, validated templates (`app/prompts.py`).
3. **Inference configuration**: document temperature / top-k / top-p choices for each task type and justify them from your Month 1 work.
4. **Baseline system**: the simplest end-to-end version (prompt + model, no RAG, no fine-tuning) that does the task acceptably. Generate samples and commit them to `samples/before_improvements.txt`.
5. **Input validation and guardrails**: length caps, schema checks, retry loops, refusal rules — the reliability toolkit from Month 1.3.

### Requirements (all unit-tested — Gate G2)

1. **Prompt templates** are parameterized and validated; every template has a test that the filled prompt satisfies the documented contract.
2. **Output schema** is enforced at the application level (parse + validate, failover on failure). JSON-in/JSON-out where the task allows.
3. **Baseline is measured**: log a seed corpus of test inputs and outputs, and record quality (LLM-as-judge score or a domain metric) and latency. This baseline is the comparison point for every later phase.
4. **Guardrails** work: malformed input returns a sane error, never a crash and never a leaked stack trace.

### Deliverable

`app/prompts.py`, `app/schema.py`, `samples/before_improvements.txt`, and a README section on your prompt/inference design. Tests for every template.

---

## Part 2: RAG, Tools, and Agents (Week 1–2)

### What to Build

Ground the product and give it capabilities, using exactly the Month 2 stack.

**RAG** (if your task benefits from retrieval):

1. **Corpus**: curate a domain document set (`data/`), chunk it, and store embeddings in Chroma with metadata.
2. **Retrieval**: embed the query, retrieve top-k, and rerank (cross-encoder or hybrid vector + BM25).
3. **Grounding**: generated answers must cite sources; verify that citations actually appear in the retrieved chunks.
4. **Quality metrics**: Hit Rate, MRR, Citation Precision. Measure the pipeline on a labeled set and report numbers in `eval_report.md`.

**Tools/Agents** (if your task benefits from actions):

1. **Function calling**: design tool schemas, route calls, handle errors and retries with backoff.
2. **ReAct-style loop**: Thought → Action → Observation with step limits, token budgets, and a guardrail that stops runaway loops.
3. **Observability**: every tool call logged with input, output, latency, and error status.

At least one of RAG or tools/agents is mandatory; a strong product will usually need both.

### Requirements (all unit-tested — Gate G2)

1. **Retrieval correctness**: tests that the expected chunk is in the top-k for labeled queries; embedding dimension and metadata filters are correct.
2. **Reranking actually reranks**: top-1 recall after rerank >= top-1 recall before rerank on your eval set.
3. **Citations verified**: a test that every citation marker in an answer maps to a retrieved chunk.
4. **Agent loop terminates**: max-steps enforcement, no infinite loops; a test that a "stuck" agent errors cleanly.

### Deliverable

`app/rag.py`, `app/agent.py`, `app/tools.py`, `data/` (corpus + download script), and tests. Report retrieval quality numbers in `eval_report.md`.

---

## Part 3: Dataset and LoRA Fine-Tuning (Week 2–3)

### What to Build

A domain dataset and a LoRA fine-tune on top of a pretrained open-weight model (Phi-2, Gemma-2B, TinyLlama). This is the Month 3 playbook applied to your product.

### Dataset curation

1. **Source**: a real, licensed domain corpus (or a labeled task dataset you build). Document the license in `data/dataset_card.md`.
2. **Cleaning**: `data/clean.py` with exact dedup, a documented quality filter, and a train/val/test split that prevents leakage. Document every threshold.
3. **Format**: convert to the instruction/chat format your base model expects; tokenize once with the base model's tokenizer; verify token counts and lengths.

### Fine-tuning (all from `finetune/`)

1. **LoRA config**: choose `r`, `alpha`, `target_modules`, `dropout`, `task_type`; justify every number against your compute budget.
2. **Training**: HF `Trainer` (as taught) with gradient accumulation, `fp16`/`bf16`, and checkpoints. Save the adapter (`save_pretrained`) — never merge into the base unless you document why.
3. **Regression testing**: before/after comparison on a fixed eval set — the tuned adapter must not regress general capability while improving the domain task.
4. **Tracking**: W&B or tensorboard curves for train/val loss; commit checkpoints and curves to the repo.

### Requirements (all unit-tested — Gate G2)

1. **Data script reproducible**: `data/download.py` + `data/clean.py` rerun to the same files from a fresh clone.
2. **Split integrity**: no document appears in more than one split (hash check in a test).
3. **Adapter loads**: `PeftModel.from_pretrained` reproduces the eval results you report.
4. **Format correct**: a test that the formatted examples match the model's expected chat template.

### Deliverable

`data/` (download.py, clean.py, dataset_card.md), `finetune/` (prep_data.py, train_lora.py, checkpoints, logs), and a training section in `eval_report.md` (loss curves, tokens, wall-clock, what broke and how you fixed it).

---

## Part 4: Evaluation (Week 3)

### The eval harness (`eval/`)

1. **LLM-as-judge** (`eval/judge.py`): score outputs 1–5 on the dimensions you define (factual accuracy, coherence, format compliance, domain usefulness). Control for the biases Month 5 teaches: position, verbosity, self-enhancement. Document the rubric.
2. **Human rating**: you plus 2 independent raters score a sample; report inter-rater agreement (Cohen's kappa) honestly.
3. **Domain benchmark**: 20 diverse, domain-specific prompts, at least 3 "trap" prompts (valid JSON, format compliance). Fixed seed, fixed temperature. Every output committed to `samples/`.
4. **A/B comparison**: your final system vs your Part 1 baseline (and, if applicable, vs the untuned base model). Statistical analysis as taught in Month 5 (t-test, effect size).
5. **Red-team**: attack your own system (prompt injection, jailbreaks, data exfiltration) and document mitigations in `eval/red_team_report.md`.

### Comparison table

| Metric | Baseline (Part 1) | Base model (untuned) | Fine-tuned (Part 3) | Final system (full stack) |
|---|---|---|---|---|
| LLM-judge score (1–5) | | | | |
| Domain metric (MRR / hit rate / accuracy) | | | | |
| Latency (100 tokens) | | | | |
| Throughput (tok/s) | | | | |
| Training cost | | | | |
| Error rate | | | | |

Write an honest tradeoff paragraph: where each configuration wins and when you would ship each in production.

### Requirements (all unit-tested — Gate G2)

1. **Judge reproducibility**: same inputs + same judge model + temperature 0 ⇒ identical scores (within tolerance).
2. **Rubric documented**: the 1–5 definitions are written down and applied consistently by all raters.
3. **Numbers are real**: every number in the comparison table is reproducible from committed code and data.

### Deliverable

`eval/` (judge.py, benchmark.py, ab_test.py), `eval/eval_report.md`, `eval/red_team_report.md`, `samples/after_improvements.txt`.

---

## Part 5: Serve the Model (Week 3–4)

### FastAPI server — required endpoints

| Endpoint | Behavior |
|---|---|
| `POST /generate` | Validated params (bounds on temperature/max_tokens/top_k). Returns output, token counts, latency, `request_id`. |
| `POST /stream` | SSE streaming of tokens as produced, reusing the prompt KV cache. TTFT must be visibly lower than buffered. |
| `GET /health` | Model + tokenizer loaded, cache warm. Version + uptime. |
| `GET /metrics` | Prometheus format. |

```python
# serve/api.py (shape only — yours must be complete, tested, production-quality)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=4096)
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=vocab_size)

@app.post("/generate")
def generate(req: GenerateRequest, request: Request):
    # validate, rate-limit, check cache (semantic, similarity >= 0.95)
    # run RAG/agent pipeline + generate; log structured audit entry; return request_id
    ...

@app.post("/stream")
async def stream(req: GenerateRequest):
    # SSE generator yielding tokens; reuse prompt KV cache
    ...
```

### Server internals

1. **KV-cache reuse**: extending a prompt reuses its cached prefix. Measure the speedup, report it.
2. **Semantic cache**: identical/near-identical prompts (cosine >= 0.95) skip the model. Report hit rate and savings.
3. **Batching**: concurrent `/generate` calls share a forward pass. Static batching is the minimum; **continuous batching** (fill freed slots) is required for distinction.
4. **Backpressure**: bounded queue; saturated => `429` + `Retry-After`, never unbounded growth.
5. **Streaming**: prove with a measurement that streaming cuts time-to-first-token.

### Load test (Gate G5)

`ops/locustfile.py`, **30-minute soak, 50 concurrent users**, mixing `/generate` and `/stream`. Commit results:

- p50/p95/p99 TTFT and TTLT
- 0 HTTP 5xx, 0 dropped requests
- **Flat memory curve** (a leak is a failure)
- requests/s throughput

### UI — Gradio or Streamlit

- Streams token-by-token
- temperature / top-k / top-p controls
- shows latency + token counts
- **edit-and-resubmit** (edit output, re-ask)
- **rating** (thumbs up/down -> audit store)

### Docker

Multi-stage `ops/Dockerfile`, runs as **non-root**, pinned base image. `docker-compose.yml` for api + ui + prometheus + grafana.

### Deliverable

`serve/` (api.py, server.py, cache.py, streaming.py, audit.py, metrics.py, ui.py, test_api.py), `ops/locustfile.py`, load-test results, `ops/Dockerfile`, `ops/docker-compose.yml`.

---

## Part 6: Observability, Safety, UX (Week 4)

### Observability

- **Structured logs**: one JSON line per request — `request_id`, user, model version, token counts, latency breakdown (preprocess/inference/postprocess), cache hit/miss, errors. No `print()`.
- **Metrics** (Prometheus): request count, latency histogram, error rate, tok/s, cache hit rate, queue depth, memory, estimated cost/request.
- **Tracing**: `api -> pipeline -> model` span chain per request.
- **Grafana dashboard**: latency percentiles, error rate, throughput, cache hit rate, cost. Screenshot committed.
- **Alerts** + `ops/incident_playbook.md` covering at least 3 scenarios (latency spike, error spike, memory leak).

### Audit trail

Every interaction (input, output, params, model version, timings, ratings) in an append-only store with a documented schema. It must be **queryable** ("show all requests from user X on this day with ratings"). Ship a query CLI or script.

### Safety & guardrails

- Input validation, rate limiting per user/IP
- Output blocklist + hard token cap
- Timeouts everywhere, sane error codes, no leaked stack traces
- "Known weaknesses" section in README with production mitigations
- Red-team findings and fixes documented in `eval/red_team_report.md`

### HITL + UX

- UI wireframes committed
- **HITL decision logic**: a confidence signal (mean token log-prob or a small auxiliary classifier) with a threshold below which outputs go to a human review queue instead of returning
- **Usability test**: 3–5 users, 3–5 tasks, silent observation, >= 3 documented findings, top-2 fixed, re-tested

---

## Part 7: Hardening & CI/CD (Week 4)

- `pyproject.toml`: ruff (lint), mypy/pyright (types), pytest — all clean, zero warnings
- **CI** (`.github/workflows/ci.yml`): on every push — lint -> type -> test -> docker build -> **eval gate** (rerun your domain benchmark; fail on > 2% regression vs the committed baseline). Green on final commit.
- **Coverage >= 80%** on app, eval, serve
- **Makefile**: `make all`, `make test`, `make serve`, `make reproduce`, `make lint`, `make loadtest`
- **README**: architecture diagram, every design decision, cost/latency table, failure log (what broke and what you learned), run instructions

---

## The Gates (Hard Pass/Fail)

Each gate is binary. Any red gate = capstone not passed until green.

| # | Gate | Pass condition |
|---|---|---|
| G1 | Reproducibility | Fresh clone + `make reproduce` rebuilds the app and reproduces your reported eval numbers. |
| G2 | Correctness | `pytest` green: prompts, RAG, agents, judge, API, cache, batching, streaming, audit. |
| G3 | Quality | Final system beats your documented baseline on the domain benchmark: >= 60% LLM-judge win rate (or an equivalent, pre-approved domain metric margin). |
| G4 | Speed | TTFT < 250 ms, >= 15 tok/s at batch 1 on the course machine (API or documented local model). |
| G5 | Stability | 30-min locust soak, 50 users: p95 < 2 s, 0 errors, flat memory. |
| G6 | Tests & CI | Coverage >= 80%; CI green on submitted commit. |
| G7 | Auditability | Every request logged with trace id; audit store queryable. |
| G8 | History | Incremental commits over >= 3 weeks. |
| G9 | Defense | Pass the 45-minute oral defense. |

---

## Stretch Goals (Distinction, up to +20)

Each verified in the defense. Points stack only if G1–G9 are green.

- Continuous batching (+5) — proven with a concurrency plot
- Multimodal integration (+4) — vision/audio input in your pipeline (Month 4 stack)
- Advanced RAG (+3) — query rewriting, HyDE, hybrid search, citation verification
- INT8/4-bit quantized inference (+3) — < 1% quality regression, measured speedup
- Multi-agent orchestration (+4) — planner + worker agents with verification
- In-production A/B testing (+3) — traffic split with statistical analysis
- Automatic model routing (+3) — small/cheap model vs large model by prompt complexity
- Custom eval metric for your domain (+3) — designed, validated, and regression-gated in CI
- Production extras (+2 each, max +4) — auth, multi-tenancy, error taxonomy

Distinction requires >= 10 bonus points and G3 beaten by a wider margin (>= 75% win rate).

---

## Grading Rubric (200 points)

| Category | Points | Notes |
|---|---|---|
| Phase 0 proposal | 10 | Depth, defensibility, risk awareness |
| Product core + prompting | 20 | Prompt system, schema, guardrails, baseline |
| RAG + tools/agents | 25 | Retrieval quality, grounding, agent correctness |
| Data + fine-tuning | 25 | Pipeline, LoRA quality, regression testing |
| Evaluation | 25 | Judge harness, human raters, A/B, red-team |
| Serving | 30 | API 8, streaming + cache 8, batching 8, UI 6 |
| Observability | 15 | Logs/metrics/tracing 8, dashboard/alerts 7 |
| UX + HITL + audit | 20 | 5 each (UX, HITL, audit, safety) |
| Hardening + CI + docs | 15 | Lint/type/CI green, README quality |
| Defense | 10 | Oral performance |
| Stretch bonus | +20 max | Section above |

Pass threshold: **>= 140/200**, all G1–G9 green, no R1–R8 violation.

---

## The Defense (45 minutes, mandatory)

- **10 min** — you present: architecture, decisions, results table, live demo.
- **25 min** — panel asks about your code. Expect:
  - "Walk us through your RAG pipeline. Why this chunk size? Where does grounding break?"
  - "Explain LoRA. Why rank 8 and alpha 16? What does the delta matrix capture?"
  - "How does your LLM-as-judge work? What biases did you control for, and how?"
  - "Where is the semantic cache keyed? Why a 0.95 cosine threshold?"
  - "Where exactly is the KV cache stored? How is it freed? Two requests share a prompt?"
  - "Which optimization gave the biggest latency win — cache, batching, or streaming? Prove it with your numbers."
  - "Show me your A/B analysis. What does the t-test actually tell you here?"
  - "What would you change with 5x more data? With 5x less compute?"
  - "Where does your system fail? What is in your known-weaknesses section?"
- **10 min** — feedback and verdict.

You may not bring notes written by anyone else. If you cannot explain a line in your repo, it is treated as not yours.

---

## Timeline (4 weeks, day-by-day)

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
| 15 | Judge harness + rubric + kappa | Phase 4 |
| 16 | 20-prompt benchmark + trap prompts | Phase 4 |
| 17 | A/B vs baseline + statistical analysis | Phase 4 |
| 18 | Red-team pass + fixes; eval_report.md | Phase 4 |
| 19 | FastAPI `/generate`, `/health`, `/metrics` | Phase 5 |
| 20 | SSE `/stream` + streaming tests | Phase 5 |
| 21 | KV-cache reuse + semantic cache | Phase 5 |
| 22 | Batching + backpressure | Phase 5 |
| 23 | UI (streaming, controls, rating, edit) | Phase 5 |
| 24 | Locust load test; fix; 30-min soak | G5 |
| 25 | Metrics + logs + tracing + Grafana + alerts | Phase 6 |
| 26 | Audit store + query tool + rate limiting | Phase 6 |
| 27 | HITL + review queue + usability test | Phase 6 |
| 28 | Fix top-2 UX issues; re-test | Phase 6 |
| 29 | CI/CD, coverage, Makefile, README, samples | Phase 7 |
| 30 | `make reproduce` clean run + gate check | G1–G8 |
| 31 | Defense prep + mock defense | Defense |
| 32 | **Submission + defense** | G9 |

---

## Hardware Requirements

| Setup | Cost | What You Can Run |
|---|---|---|
| CPU-only laptop | $0 | API models + 2B-class LoRA (slow but fine for tests) |
| Colab Free (T4, 16GB) | $0 | LoRA fine-tune of 2B-class models, fast eval |
| Colab Pro (A100, 40GB) | ~$10/mo | LoRA fine-tune of larger open-weight models |
| RTX 3090/4090 (24GB) | $0 (owned) | Comfortable LoRA + local serving |
| RunPod / Lambda Labs | ~$0.50/hr | Anything you need, no queue |

The API-model route works for everything except the LoRA phase, which needs a small open-weight model on free Colab. Start early — Gate G3 punishes late starts harder than weak hardware.

---

## What You Actually Put on Your Resume

**Portfolio Project: Built and shipped a production-grade [domain] GenAI product**

- Designed a prompt system with validated schemas and guardrails, and measured a baseline with LLM-as-judge
- Built a RAG pipeline (embeddings, retrieval, reranking, citation grounding) with [Hit Rate X%, MRR Y] on a labeled set
- Fine-tuned [Phi-2/Gemma-2B/TinyLlama] with LoRA on [dataset] — improving domain quality by [Z]% by judge score without regressing general ability
- Evaluated with an LLM-as-judge harness (Cohen's kappa [κ]), a 20-prompt domain benchmark, and an A/B test against baseline
- Deployed as a FastAPI server with SSE streaming, semantic caching, batching, Prometheus monitoring, and a Gradio UI in Docker — passing a 50-concurrent-user, 30-minute load test

**Interview talking point:** "Here's a table comparing my baseline, my fine-tuned model, and my full system. LoRA gave me [Z]% domain improvement, RAG gave me citation-grounded answers, and caching/batching cut p95 latency by [W]%. That's the tradeoff stack I want to help your team optimize."

---

## Submission Checklist

- [ ] Proposal approved (Phase 0)
- [ ] `make all` and `make reproduce` work from a fresh clone (G1)
- [ ] `pytest` green, coverage >= 80% (G2, G6)
- [ ] Domain benchmark beats baseline by >= 60% judge win rate (or approved metric) (G3)
- [ ] TTFT < 250 ms, >= 15 tok/s (G4)
- [ ] 30-min soak: p95 < 2 s, 0 errors, flat memory (G5)
- [ ] CI green on final commit (G6)
- [ ] Audit store queryable; every request traced (G7)
- [ ] Incremental git history (G8)
- [ ] README with architecture, decisions, tables, failure log
- [ ] `eval_report.md`, `ux_report.md`, `incident_playbook.md`, `optimization_report.md`, `slo_report.md`, `red_team_report.md`
- [ ] Defense booked and passed (G9)

---

## Summary

This capstone covers the full stack you were taught: product design and prompting (Month 1), RAG and agents (Month 2), LoRA fine-tuning (Month 3), evaluation and safety (Month 5), and serving, observability, and hardening (Month 6). The gates (G1–G9) and the defense ensure the work is real and understood. When you finish, you have a deployable GenAI product you built and shipped yourself, end-to-end.
