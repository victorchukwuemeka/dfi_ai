# Capstone: Train a Language Model From Scratch

The final project of the course. You will build a working language model end-to-end: a byte-level BPE tokenizer, a decoder-only transformer, a training loop, an evaluation harness, and a production server. No starter code is provided.

The full contract (rules, gates, rubric, defense) is in `month_6/06_deployment_capstone.md`. This file is the technical specification.

## Deliverable

A small GPT-style transformer (~10–50M params) that you:

1. Tokenize with your own byte-level BPE (no `tokenizers`, no `tiktoken`)
2. Implement in pure PyTorch (no HF Trainer, no pretrained weights)
3. Train from random initialization on a domain-specific dataset
4. Evaluate with perplexity and generation benchmarks against real baselines
5. Serve through a FastAPI server with streaming, caching, batching, monitoring, and a web UI
6. Prove it works: unit tests, CI, load tests, and a live defense

All code in one GitHub repo with a README covering every design decision.

## Rules

A violation of any rule is an automatic capstone failure (grade 0).

| # | Rule |
|---|------|
| R1 | The model must be pure PyTorch. No HF `Trainer`, no `transformers` model classes, no `keras`, no `timm`. |
| R2 | Training must start from random initialization. Loading any pretrained weights is not allowed. |
| R3 | The tokenizer must be a byte-level BPE you implement yourself. `tokenizers` and `tiktoken` are not allowed. You may use only `regex` for pre-tokenization. |
| R4 | Every component must have unit tests you wrote: tokenizer, RoPE, GQA, KV cache, sampling, training loop, API. |
| R5 | The project must run from a fresh clone on the course machine via the provided Makefile. |
| R6 | Git history must show incremental development across the project. A one-shot "everything" commit will be investigated. |
| R7 | Tests and the demo must run on CPU. |
| R8 | Everything lives in the repo and is referenced from the README. |

## Tech Stack

| Layer | Choice | Constraint |
|---|---|---|
| Framework | PyTorch | No HF Trainer — write the loop yourself |
| Tokenizer | Custom byte-level BPE | No `tokenizers` / `tiktoken` |
| Architecture | Decoder-only transformer with RoPE, RMSNorm, SwiGLU, GQA | Implement every component yourself |
| Training | Single GPU (RTX 3090/4090 or T4 colab), mixed precision | Full loop hand-written |
| Data | Domain corpus you curate (code, medical, legal, finance — pick one) | 200 MB–2 GB, cleaned and deduplicated |
| Serving | FastAPI + Gradio/Streamlit + Docker | SSE streaming, caching, batching, metrics |
| Tracking | W&B or local tensorboard | Log everything; publish your curves |
| Quality | Perplexity + generation benchmarks + LLM-as-judge | Head-to-head vs a fine-tuned model and an API |

---

## Part 1: Build the Tokenizer (Week 1)

### What to Build

A byte-level BPE tokenizer trained on your corpus, written from scratch.

```
Corpus: "the cat sat on the mat"
Step 1:  pre-tokenize with the GPT-2 regex -> ["the", " cat", " sat", " on", " the", " mat"]
Step 2:  split to bytes, count pairs -> ("t","h"): 2, ("h","e"): 2, (" ","c"): 1 ...
Step 3:  merge most frequent pair -> ("t","h") -> "th"
Step 4:  repeat until vocab_size is reached
```

### Requirements (all unit-tested — Gate G2)

1. **Pre-tokenization** with the GPT-2 regex pattern. Your splits must match reference behavior on a torture test: unicode, emoji, code, contractions, multi-byte characters.
2. **Byte-level base**: initial vocab is bytes 0–255, so any UTF-8 input round-trips.
3. **Merge training** to a configurable `vocab_size`, deterministic tie-break, saved `merges` list.
4. **Special tokens** (`<|endoftext|>` etc.) that are never split and encode/decode correctly.
5. **Round-trip guarantee**: `decode(encode(text)) == text` for 100% of a held-out corpus.
6. **Determinism**: same corpus + seed => byte-identical vocab and merges.
7. **Efficiency**: tokenizing your full corpus in under ~30 minutes. Report tok/s.

### Implementation

```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def pre_tokenize(self, text: str) -> list[str]:
        # GPT-2 regex split. Document the pattern you choose.
        ...

    def train(self, corpus: list[str]) -> None:
        # 1. Pre-tokenize
        # 2. Byte vocab 0-255
        # 3. Pair counting. Do NOT rescan the corpus every iteration.
        #    Maintain pair frequencies incrementally as merges happen.
        # 4. Merge until vocab_size. Deterministic tie-break, documented.
        # 5. Save merge rules.
        ...

    def encode(self, text: str) -> list[int]:
        # Apply merge rules in ranked order. Round-trip must be exact.
        ...

    def decode(self, ids: list[int]) -> bytes:
        # Map ids to bytes, join. Should never need errors="replace".
        ...
```

### Checked in review

- The merge loop must be reasonably efficient. If it rescans the corpus per merge it will not pass the 30-minute bar. Be able to explain the data structures you used.
- The vocab/merges files must be provably produced by your `train()` — no `tokenizers` artifacts in the repo.
- Edge cases: empty string, whitespace-only input, lone surrogates, long repeated runs.

### Deliverable

`tokenizer/bpe.py`, `tokenizer/files/` (committed), `tokenizer/test_bpe.py`, and a README section on your data structures and complexity.

---

## Part 2: Implement the Transformer (Week 1–2)

### Architecture

A GPT-style decoder-only transformer:

```
Input tokens (B, T)
    |
Token Embedding (vocab_size -> d_model)
    |
    +-- [Transformer Block x N] --------------------+
    |                                                |
    |   RMSNorm -> Grouped Query Attention (RoPE)    |
    |        -> Residual + RMSNorm -> SwiGLU FFN     |
    |        -> Residual                             |
    |                                                |
    +------------------------------------------------+
    |
RMSNorm
    |
Linear (d_model -> vocab_size)  [weight-tied with token embedding]
    |
Logits (B, T, vocab_size)
```

### Correctness tests (mandatory — Gate G2)

| Test | What it proves |
|---|---|
| **RoPE rotation** | `apply_rope` at two positions equals a hand-written rotation-matrix reference. |
| **Position sensitivity** | Attention scores between identical-token pairs differ across positions. |
| **Cache equivalence** | Generation with KV cache produces byte-identical tokens to generation without, for sequences >= 64 tokens. |
| **GQA shapes** | `n_heads % n_kv_heads == 0`; grouped broadcast correct. |
| **Weight tying** | `lm_head.weight is token_embedding.weight` (same tensor object). |
| **Masking** | Altering token `t+1` does not change logits at position `t`. |
| **Gradient check** | `torch.autograd.gradcheck` passes on a tiny model for RMSNorm, attention, SwiGLU, block. |
| **Sampling** | temperature=0 -> argmax; top-k=1 -> argmax; top-p=1.0 -> no-op; seeded -> reproducible. |

### Component requirements

**RMSNorm**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x / sqrt(mean(x^2) + eps) * weight
        ...
```

**RoPE (Rotary Position Embeddings)** — position information is baked into attention; no learned positional embeddings.

```python
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    # freqs = 1 / (theta ^ (2i / dim)) for i in 0..dim/2
    ...

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, T, n_heads, head_dim) — rotate half the dims, concatenate.
    # Shape handling is where bugs hide. Test against a reference.
    ...
```

**Grouped Query Attention** — with a growable KV cache for inference (supporting prefix reuse) and no cache for training.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        # n_heads % n_kv_heads == 0. Each KV head serves n_heads // n_kv_heads query heads.
        ...

    def forward(self, x, freqs, mask=None, use_cache=False):
        # Q, K, V projections
        # RoPE on Q, K
        # Grouped broadcast of K, V
        # Causal scaled dot-product attention
        # Cache: append new K, V; attend over cache. MUST equal no-cache output.
        ...
```

**SwiGLU FFN**

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float = 8 / 3):
        # (x @ W1) * silu(x @ W2) @ W3
        ...
```

**Transformer Block** — pre-norm residual: `x = x + attn(norm1(x))`, `x = x + ffn(norm2(x))`.

**GPT Model**

```python
class GPT(nn.Module):
    def __init__(self, config: dict):
        # vocab_size, d_model, n_layers, n_heads, n_kv_heads, max_seq_len
        ...
        self.token_embedding.weight = self.lm_head.weight  # weight tying — same object

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        ...

    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, seed=None, on_token=None):
        # Autoregressive loop with KV cache.
        # temperature, top-k, top-p, deterministic under seed.
        # on_token callback streams tokens one at a time (used by the server).
        ...
```

### Design rationale to be able to explain

- Why RMSNorm instead of LayerNorm
- Why RoPE instead of learned position embeddings (extrapolation)
- Why GQA instead of MHA (inference speed, minor quality loss)
- Why SwiGLU instead of ReLU/GELU
- Why weight tying (fewer parameters, acts as regularization)
- Why the KV cache is growable and why it must exactly match no-cache generation

### Deliverable

`model/` — self-contained transformer with `forward()` and `generate()`, plus `model/test_model.py` covering every test in the table above.

---

## Part 3: Train the Model (Week 2–3)

### Training Setup

| Hyperparameter | Typical Value (10M model) | Typical Value (50M model) |
|---|---|---|
| d_model | 256 | 512 |
| n_layers | 6 | 12 |
| n_heads | 8 | 8 |
| n_kv_heads | 4 | 4 |
| max_seq_len | 512 | 1024 |
| vocab_size | 8192 | 16384 |
| Total params | ~10M | ~50M |
| Batch size | 64 | 32 |
| Learning rate | 3e-4 | 3e-4 |
| Warmup steps | 1000 | 2000 |
| Total tokens | ~500M | ~2B |

Justify every number against your actual compute budget.

### Dataset curation

Pick ONE domain and own it:

| Domain | Data Source | Why |
|---|---|---|
| Code | The Stack v2 (subset), GitHub repos | Employers like code models |
| Medical | PubMed abstracts, clinical notes | High-value specialization |
| Legal | CourtListener, Pile of Law | Structured reasoning |
| Finance | SEC filings, earnings transcripts | Numeracy + domain understanding |
| Your niche | Whatever you know deeply | You can evaluate better |

Curation is not "download and go." You must:

1. **Download reproducibly** (`data/download.py` — pinned URLs, licenses documented).
2. **Deduplicate** (exact + a documented fuzzy method).
3. **Filter quality** (avg token length, special-char ratio, newline ratio — thresholds documented and justified).
4. **Split** train/val/test with the same distribution — document how you prevent leakage.
5. **Tokenize once** to binary chunks (`train.bin` / `val.bin` / `test.bin`, raw `uint16`/`uint32` ids).
6. **Stream** — the dataset must never be materialized fully in RAM (`mmap` or chunked reads).

### The training loop — write every line yourself

```python
def train(model, dataloader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.1
    )
    # Decoupled weight decay: no decay on norms/biases/embeddings.
    # Parametrize which params get decay.

    scheduler = CosineScheduleWithWarmup(optimizer, config["warmup"], config["total_steps"])
    # Implement the formula yourself.

    scaler = torch.amp.GradScaler()
    accumulation_steps = config["grad_accum"]

    for step, batch in enumerate(dataloader):
        x, y = batch[:, :-1], batch[:, 1:]

        with torch.amp.autocast(device_type="cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()           # scale
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)          # unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip
            scaler.step(optimizer)              # step
            scaler.update()                     # update scale
            optimizer.zero_grad()
        scheduler.step()

        # Log: step, loss, lr, grad norm, tokens/sec, memory.
        # NaN/Inf guard: detect, log, recover without corrupting the run.
        # Checkpoint: latest.pt + rolling history. Resume = exact restore of
        #   step counter, optimizer, scheduler, AND random state (seed).
        # Every 1,000 steps: generate a sample, commit to samples/.
```

Required in your loop, each provable from logs:

1. AdamW with decoupled weight decay, default 0.1
2. Cosine schedule with linear warmup — your own implementation
3. Mixed precision in the exact order: scale -> backward -> unscale -> clip -> step -> update
4. Gradient clipping to a justified norm
5. Gradient accumulation for large effective batch on a small GPU
6. Checkpointing + exact resume
7. Seeded determinism
8. NaN/Inf detection and recovery
9. Full logging (step, loss, lr, grad norm, tok/s, memory)

### The quality gate

The course publishes a **hidden reference checkpoint**: a model trained with the same budget (same tokens, similar hyperparameter space) on the same processed dataset, published only as a val perplexity number. You must either:

- Beat it by **>= 10% relative**, or
- Tie within **3%** at <= 80% of its parameters (win on efficiency, not size).

If your tokenizer, model, or loop has any silent bug, you land above the reference and fail. The reference is published only in the final week, so you cannot reverse-engineer it. The 10% typically comes from data quality (dedup, filtering), not architecture.

### Rationale to be able to explain

- Why cosine schedule with warmup beats constant LR
- Why weight decay 0.1 for transformers
- Why gradient clipping prevents loss spikes
- Why mixed precision gives ~2x speedup with no quality loss
- How to pick batch size for your memory budget
- Why you scale tokens, not steps
- Why the AMP order (scale -> backward -> unscale -> clip -> step -> update) is the only correct one

### Deliverable

`train/train.py`, `train/data.py`, `data/clean.py`, `data/download.py`, checkpoints, training curves, and a training report in `eval_report.md`.

---

## Part 4: Evaluate the Model (Week 3)

### Perplexity

```python
def compute_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)
```

Report on the **test split** (untouched by training). Report loss and perplexity with token counts, and explain what the perplexity value means for your domain.

### Generation benchmarks

Create 20 diverse, domain-specific prompts (at least 3 "trap" prompts requiring format compliance — valid JSON, compilable code). For each:

- Generate with fixed seed, temperature 0.7.
- Score 1–5 on **factual accuracy**, **coherence**, **format compliance** — by you, by 2 independent raters, and by an LLM judge.
- Report inter-rater agreement (Cohen's kappa) honestly.

### Comparison baseline

Compare against:

- **Your fine-tuned model from Month 3** (domain-tuned LLM)
- **GPT-4o-mini** (API — the standard)

| Metric | Your Model (10M) | Your Model (50M) | Fine-tuned LLM | GPT-4o-mini |
|---|---|---|---|---|
| Perplexity | | | N/A (API) | N/A (API) |
| Inference latency (100 tok) | | | | |
| Throughput (tok/s) | | | | |
| Model size | | | | |
| Training cost | | | | |
| Generation quality (1–5) | | | | |

Write an honest tradeoff paragraph: where your model wins, where it loses, and when you would choose each option in production.

### Rationale to be able to explain

- Perplexity is necessary but not sufficient — it misses generation quality
- Small models are surprisingly capable at their size
- Training from scratch vs fine-tuning vs API — when each is right
- The pareto frontier of quality vs cost — and where you sit on it

### Deliverable

`eval/` (perplexity.py, benchmark.py, judge.py), `eval/eval_report.md`, `samples/after_training.txt`.

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
    # generate with KV cache; log structured audit entry; return request_id
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
- **Tracing**: `api -> batch -> model` span chain per request.
- **Grafana dashboard**: latency percentiles, error rate, throughput, cache hit rate, cost. Screenshot committed.
- **Alerts** + `ops/incident_playbook.md` covering at least 3 scenarios (latency spike, error spike, memory leak).

### Audit trail

Every interaction (input, output, params, model version, timings, ratings) in an append-only store with a documented schema. It must be **queryable** ("show all requests from user X on this day with ratings"). Ship a query CLI or script.

### Safety & guardrails

- Input validation, rate limiting per user/IP
- Output blocklist + hard token cap
- Timeouts everywhere, sane error codes, no leaked stack traces
- "Known weaknesses" section in README with production mitigations

### HITL + UX

- UI wireframes committed
- **HITL decision logic**: a confidence signal (mean token log-prob or auxiliary classifier) with a threshold below which outputs go to a human review queue instead of returning
- **Usability test**: 3–5 users, 3–5 tasks, silent observation, >= 3 documented findings, top-2 fixed, re-tested

---

## Part 7: Hardening & CI/CD (Week 4)

- `pyproject.toml`: ruff (lint), mypy/pyright (types), pytest — all clean, zero warnings
- **CI** (`.github/workflows/ci.yml`): on every push — lint -> type -> test -> docker build -> **eval gate** (test-split perplexity; fail on > 2% regression). Green on final commit.
- **Coverage >= 80%** on tokenizer, model, eval, serve
- **Makefile**: `make all`, `make test`, `make serve`, `make reproduce`, `make lint`, `make loadtest`
- **README**: architecture diagram, every design decision, cost/latency table, failure log (what broke and what you learned), run instructions

---

## The Gates (Hard Pass/Fail)

Each gate is binary. Any red gate = capstone not passed until green.

| # | Gate | Pass condition |
|---|---|---|
| G1 | Reproducibility | Fresh clone + `make reproduce` reproduces your best within 0.05 val loss. |
| G2 | Correctness | `pytest` green: RoPE, cache equivalence, masking, gradcheck, tokenizer round-trip. |
| G3 | Quality | Val perplexity beats hidden reference by >= 10% rel, or ties within 3% at <= 80% of its params. |
| G4 | Speed | TTFT < 250 ms, >= 15 tok/s at batch 1 on the course machine (GPU or documented CPU). |
| G5 | Stability | 30-min locust soak, 50 users: p95 < 2 s, 0 errors, flat memory. |
| G6 | Tests & CI | Coverage >= 80%; CI green on submitted commit. |
| G7 | Auditability | Every request logged with trace id; audit store queryable. |
| G8 | History | Incremental commits over >= 3 weeks. |
| G9 | Defense | Pass the 45-minute oral defense. |

---

## Stretch Goals (Distinction, up to +20)

Each verified in the defense. Points stack only if G1–G9 are green.

- Continuous batching (+5) — proven with a concurrency plot
- Speculative decoding / draft model (+4) — speedup at identical output quality
- Flash attention / `scaled_dot_product_attention` (+3) — with equivalence test
- INT8 quantization (+3) — < 1% perplexity regression, measured speedup
- DDP / FSDP distributed training (+4) — correct scaling curve
- DPO/PPO-style preference tuning (+4) — measured win-rate vs base
- Custom CUDA kernel (+5) — fused RMSNorm+RoPE or KV-append
- Automatic model routing (+3) — your model vs API by prompt complexity
- Production extras (+2 each, max +4) — auth, multi-tenancy, error taxonomy

Distinction requires >= 10 bonus points and G3 beaten by >= 15% relative.

---

## Grading Rubric (200 points)

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
| Stretch bonus | +20 max | Section above |

Pass threshold: **>= 140/200**, all G1–G9 green, no R1–R8 violation.

---

## The Defense (45 minutes, mandatory)

- **10 min** — you present: architecture, decisions, results table, live demo.
- **25 min** — panel asks about your code. Expect:
  - "Walk us through your merge loop. Worst-case complexity?"
  - "Derive the RoPE formula. Why does it extrapolate?"
  - "Show the tensor shapes through `apply_rope`."
  - "Where is the KV cache stored? How is it freed? Two requests share a prompt?"
  - "Why does your loss curve hump at step 4,000?"
  - "Why that cache-equivalence tolerance? What does a bug look like numerically?"
  - "What would you change with 5x compute? With 5x less data?"
  - "Your model or GPT-4o-mini for your domain — which ships, and why?"
- **10 min** — feedback and verdict.

You may not bring notes written by anyone else. If you cannot explain a line in your repo, it is treated as not yours.

---

## Timeline (4 weeks, day-by-day)

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
| 11 | `generate()` + sampling tests | Phase 2 |
| 12 | Data clean: dedup + filters + split + `.bin` | Phase 3 |
| 13 | Train loop v1 (AMP, clip, checkpoint, resume) | Phase 3 |
| 14 | Scheduler + grad accumulation + NaN guard | Phase 3 |
| 15 | First short run; fix bugs | Phase 3 |
| 16 | Main training run; start eval harness | Phase 3/4 |
| 17 | Perplexity + 20-prompt benchmark + raters | Phase 4 |
| 18 | Baseline table; eval_report.md | Phase 4 |
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

| Setup | Cost | What You Can Train |
|---|---|---|
| Colab Free (T4, 16GB) | $0 | ~10M params, ~500M tokens |
| Colab Pro (A100, 40GB) | ~$10/mo | ~50M params, ~2B tokens |
| RTX 3090/4090 (24GB) | $0 (owned) | ~50M params, solid |
| RunPod / Lambda Labs | ~$0.50/hr | ~350M params + flash attention |

The free Colab route works, but start early — Gate G3 punishes late starts harder than weak hardware.

---

## What You Actually Put on Your Resume

**Portfolio Project: Trained a 50M-parameter GPT-style language model from scratch**

- Built a byte-level BPE tokenizer (8K–16K vocab) from scratch, trained on 2GB of [domain] text
- Implemented a decoder-only transformer with RoPE, GQA, RMSNorm, SwiGLU, and a KV cache in pure PyTorch
- Trained from random init on a single GPU for [X] hours — val perplexity [Y], beating the course reference by [Z]%
- Evaluated head-to-head against a fine-tuned LLM and GPT-4o-mini across quality, latency, and cost axes
- Deployed as a FastAPI server with SSE streaming, semantic caching, batching, Prometheus monitoring, and a Gradio UI in Docker — passing a 50-concurrent-user, 30-minute load test

**Interview talking point:** "Here's a table comparing my 50M-param model against GPT-4o-mini. Mine is 30x faster and 1000x cheaper per inference call. It's worse on general knowledge, but on my domain it's within 15% of GPT-4o-mini's quality at 0.1% of the cost. That's the tradeoff I want to help your team optimize."

---

## Submission Checklist

- [ ] Proposal approved (Phase 0)
- [ ] `make all` and `make reproduce` work from a fresh clone (G1)
- [ ] `pytest` green, coverage >= 80% (G2, G6)
- [ ] Val perplexity beats reference by >= 10% (or efficiency tie) (G3)
- [ ] TTFT < 250 ms, >= 15 tok/s (G4)
- [ ] 30-min soak: p95 < 2 s, 0 errors, flat memory (G5)
- [ ] CI green on final commit (G6)
- [ ] Audit store queryable; every request traced (G7)
- [ ] Incremental git history (G8)
- [ ] README with architecture, decisions, tables, failure log
- [ ] `eval_report.md`, `ux_report.md`, `incident_playbook.md`
- [ ] Defense booked and passed (G9)

---

## Summary

This capstone covers the full stack: tokenizer, architecture, training, evaluation, serving, observability, and hardening. The gates (G1–G9) and the defense ensure the work is real and understood. When you finish, you have a deployable model you built yourself, end-to-end.
