# Capstone: Train a Language Model From Scratch

## Why This Capstone

Every GenAI course has you call APIs. That makes you a *user*, not an *engineer*. This capstone makes you build the thing itself — a working language model, trained from scratch on your own data, served through a real interface.

After this, when an interviewer asks "how does GPT work?", you don't recite the Vaswani paper. You say "I built one."

---

## The Deliverable

A **small GPT-style transformer** (~10–50M params) that you:
1. Tokenize with your own BPE tokenizer
2. Implement in pure PyTorch (no HF Trainer)
3. Train on a domain-specific dataset
4. Evaluate with perplexity + generation benchmarks
5. Serve through a local web UI or API

All code in one GitHub repo with a README that walks through every design decision.

---

## Why This Gets You Hired

| What You Build | What It Signals |
|---|---|
| BPE tokenizer from scratch | You understand the input pipeline that every LLM depends on |
| Transformer implementation (attention, FFN, RoPE, RMSNorm) | You know the architecture, not just how to call `model.generate()` |
| Training loop (forward, backward, optimizer, scheduler, gradient clipping) | You can train models, not just fine-tune them |
| Perplexity + generation evaluation | You can measure quality, not just vibe-check |
| Web UI / API | You ship things |

---

## Tech Stack

| Layer | Choice |
|---|---|
| Framework | PyTorch (no HF Trainer — write the loop yourself) |
| Tokenizer | Custom BPE (build it or use `tokenizers` library — understand every line) |
| Architecture | Decoder-only transformer (GPT-style) with RoPE, RMSNorm, SwiGLU |
| Training | Single GPU (RTX 3090/4090 or T4 colab), mixed precision |
| Data | Domain corpus you curate (code, medical, legal, finance — pick one) |
| Serving | FastAPI + Gradio or Streamlit |
| Tracking | W&B or local tensorboard |

---

## Part 1: Build the Tokenizer

### What to Build

A Byte-Pair Encoding tokenizer trained on your corpus.

```
Corpus: "the cat sat on the mat"
Step 1:  split into chars -> ["t","h","e"," ","c","a","t"," ","s","a","t"," ","o","n"," ","t","h","e"," ","m","a","t"]
Step 2:  count pairs -> ("t","h"): 2, ("h","e"): 2, (" ","c"): 1 ...
Step 3:  merge most frequent pair -> ("t","h") -> "th"
Step 4:  repeat until vocab_size is reached
```

### Implementation

```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def train(self, corpus: list[str]):
        # 1. Pre-tokenize with regex (GPT-2 pattern)
        # 2. Initialize vocab with bytes 0-255
        # 3. Count pairs, merge most frequent, repeat
        # 4. Save merge rules
        pass

    def encode(self, text: str) -> list[int]:
        # Apply merge rules in order
        pass

    def decode(self, ids: list[int]) -> str:
        # Map ids back to bytes, join
        pass
```

### What to Learn

- Why BPE over word-level or character-level tokenization
- How `bytes` as base units handles any input (unicode, code, etc.)
- Why vocab size matters for model capacity vs efficiency
- Why GPT-2's regex pre-tokenization splits on whitespace/punctuation

### Deliverable

`tokenizer.py` with `train()`, `encode()`, `decode()` + trained tokenizer files.

---

## Part 2: Implement the Transformer

### Architecture

Build a GPT-style decoder-only transformer with these components:

```
Input tokens (B, T)
    |
Token Embedding (vocab_size -> d_model)
    |
Positional Encoding (RoPE — applied in attention)
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
Linear (d_model -> vocab_size)
    |
Logits (B, T, vocab_size)
```

### Implementation Steps

**Step 1: RMSNorm**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        pass
```

**Step 2: RoPE (Rotary Position Embeddings)**

Apply rotation to query and key vectors based on position. No learned position embeddings — the position information is baked into the attention computation itself.

```python
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    # freqs = 1 / (theta ^ (2i / dim))  for i in 0..dim/2
    # Apply cos/sin rotations per position
    pass

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, T, n_heads, head_dim)
    # Rotate half the dims, cat with rotated other half
    pass
```

**Step 3: Grouped Query Attention**

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        # n_heads query heads, n_kv_heads key/value heads (n_heads % n_kv_heads == 0)
        # Each KV head serves n_heads // n_kv_heads query heads
        pass

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor):
        # Q, K, V projections
        # Apply RoPE to Q, K
        # Reshape for GQA: expand K, V to match Q heads
        # Scaled dot-product attention
        pass
```

**Step 4: SwiGLU FFN**

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 8/3):
        # SwiGLU: (x @ W1) * silu(x @ W2) @ W3
        # hidden_dim = int(d_model * hidden_mult * 2 / 3) * 3
        # Standard FFN has one hidden projection; SwiGLU has three
        pass
```

**Step 5: Transformer Block**

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, freqs, mask):
        x = x + self.attention(self.norm1(x), freqs, mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

**Step 6: GPT Model**

```python
class GPT(nn.Module):
    def __init__(self, config: dict):
        # config: vocab_size, d_model, n_layers, n_heads, n_kv_heads, max_seq_len
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        # Tie embedding and lm_head weights
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) token indices
        # Compute RoPE freqs
        # Token embed -> blocks -> norm -> lm_head
        pass

    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        # Autoregressive generation loop
        # Apply temperature, optional top-k
        # Return generated tokens
        pass
```

### What to Learn

- Why RMSNorm instead of LayerNorm (speed, comparable quality)
- Why RoPE instead of learned position embeddings (extrapolation to longer sequences)
- Why GQA instead of MHA (much faster inference, marginal quality loss)
- Why SwiGLU instead of ReLU/GELU (better performance at same compute budget)
- Why weight tying (reduces parameter count, acts as regularization)

### Deliverable

`model.py` — self-contained transformer implementation with `forward()` and `generate()`.

---

## Part 3: Train the Model

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

### Training Loop

```python
def train(model, dataloader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup"], config["total_steps"])

    for step, batch in enumerate(dataloader):
        # batch: (B, T) token IDs
        x, y = batch[:, :-1], batch[:, 1:]

        logits = model(x)                    # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}: loss {loss.item():.4f}, lr {scheduler.get_last_lr()[0]:.6f}")
            # Log to W&B or local file
```

### Dataset Curation

Pick ONE domain and own it:

| Domain | Data Source | Why |
|---|---|---|
| Code | The Stack v2 (subset), GitHub repos | Employers love code models |
| Medical | PubMed abstracts, clinical notes | High-value domain, shows specialization |
| Legal | CourtListener, Pile of Law | Structured domain reasoning |
| Finance | SEC filings, earnings transcripts | Numeracy + domain understanding |
| Your niche | Whatever you know deeply | You can evaluate better |

Curation process:
1. Collect raw text (200MB-2GB depending on compute)
2. Deduplicate (exact + fuzzy)
3. Filter quality (heuristic: avg token length, special char ratio, perplexity filter)
4. Split into train/val/test
5. Tokenize and save as binary chunks (.bin or .npy)

### Mixed Precision Training

```python
scaler = torch.amp.GradScaler()

for batch in dataloader:
    with torch.amp.autocast(device_type="cuda"):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### What to Learn

- Why cosine schedule with warmup works better than constant LR
- Why weight decay of 0.1 is standard for transformers
- Why gradient clipping prevents loss spikes
- Why mixed precision gives ~2x speedup with no quality loss
- How to pick batch size for your GPU memory budget
- Why you need to scale tokens, not steps (3x more tokens = better model)

### Deliverable

`train.py` — full training script + `checkpoints/latest.pt`

---

## Part 4: Evaluate the Model

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

**Target perplexity on validation set** (domain-dependent, but roughly):
- Random: ~vocab_size (e.g., 8192)
- Untrained model: ~vocab_size
- After 100M tokens: ~100-200
- After 500M tokens: ~50-100
- Good small model: ~30-50

### Generation Benchmarks

Create a test set of 20 diverse prompts. Generate outputs and rate them:
- **Factual accuracy** (domain-specific)
- **Coherence** (does it stay on topic?)
- **Format compliance** (if code, does it parse?)

### Comparison Baseline

Compare against:
- **Your fine-tuned model from Month 3** (domain-tuned LLM)
- **GPT-4o-mini** (API — the standard)

Show a table:

| Metric | Your Model (10M) | Your Model (50M) | Fine-tuned LLM | GPT-4o-mini |
|---|---|---|---|---|
| Perplexity | 45.2 | 28.7 | N/A (API) | N/A (API) |
| Inference latency (100 tok) | 12ms | 28ms | 150ms | 400ms |
| Model size | 40MB | 200MB | 7GB | N/A |
| Training cost | $2 (colab) | $10 (colab) | $5 (LoRA) | $0 |
| Generation quality (1-5) | 2.5 | 3.5 | 4.0 | 4.5 |

This table is the **most impressive thing in your portfolio**. It shows you understand the entire quality-speed-cost landscape.

### What to Learn

- Perplexity is useful but doesn't capture generation quality
- Small models are shockingly capable for their size
- Training from scratch vs fine-tuning vs API — when each makes sense
- The pareto frontier of quality vs cost

### Deliverable

`eval.py` + `eval_report.md` with perplexity, generation samples, and comparison table.

---

## Part 5: Serve the Model

### FastAPI + Gradio

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50

@app.post("/generate")
def generate(req: GenerateRequest):
    tokens = tokenizer.encode(req.prompt)
    input_ids = torch.tensor([tokens])
    output_ids = model.generate(input_ids, req.max_tokens, req.temperature, req.top_k)
    output_text = tokenizer.decode(output_ids[0].tolist())
    return {"output": output_text}

# Run: uvicorn api:app --reload
```

```python
# app.py — Gradio UI
import gradio as gr

def generate_fn(prompt, temp, max_tokens):
    response = requests.post("http://localhost:8000/generate", json={
        "prompt": prompt, "max_tokens": max_tokens, "temperature": temp
    })
    return response.json()["output"]

iface = gr.Interface(fn=generate_fn, inputs=["text", "slider", "number"], outputs="text")
iface.launch()
```

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn gradio
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### What to Learn

- Model serving != model training (different infra concerns)
- Batching, caching, and concurrency for inference
- Model serialization (torch.save, torch.jit.script, ONNX export)
- Why you'd use vLLM/TGI in production but your own server is fine for a demo

### Deliverable

`api.py` + `app.py` + `Dockerfile` — deployable model server.

---

## The Full Deliverable Checklist

```
capstone/
├── README.md              # Design decisions, results, lessons learned
├── tokenizer.py           # BPE tokenizer train/encode/decode
├── tokenizer_files/       # Trained merge rules + vocab
├── model.py               # Full transformer implementation
├── train.py               # Training loop + mixed precision
├── config.py              # Hyperparameters
├── data/                  # Dataset curation scripts
│   ├── download.py
│   ├── preprocess.py
│   └── train.bin / val.bin / test.bin
├── eval.py                # Perplexity + generation benchmarks
├── eval_report.md         # Results vs baselines
├── checkpoints/
│   └── latest.pt
├── api.py                 # FastAPI inference server
├── app.py                 # Gradio/Streamlit UI
├── Dockerfile
├── requirements.txt
└── samples/               # Generated output examples
    ├── before_training.txt
    └── after_training.txt
```

---

## What You Actually Put on Your Resume

**Portfolio Project: Trained a 50M-parameter GPT-style language model from scratch**

- Built a BPE tokenizer (8K vocab) trained on 2GB of [domain] text
- Implemented a decoder-only transformer with RoPE, GQA, RMSNorm, SwiGLU in pure PyTorch
- Trained on a single GPU for [X] hours — achieved perplexity [Y] on held-out validation
- Evaluated against fine-tuned LLMs and GPT-4o-mini across quality, latency, and cost axes
- Deployed as a FastAPI server with Gradio UI in a Docker container

**Interview talking point:** "Here's a table comparing my 50M-param model against GPT-4o-mini. Mine is 30x faster and 1000x cheaper per inference call. It's worse on general knowledge, but on my domain it's within 15% of GPT-4o-mini's quality at 0.1% of the cost. That's the tradeoff I want to help your team optimize."

---

## Hardware Requirements

| Setup | Cost | What You Can Train |
|---|---|---|
| Colab Free (T4, 16GB) | $0 | ~10M params, ~500M tokens |
| Colab Pro (A100, 40GB) | ~$10/mo | ~50M params, ~2B tokens |
| RTX 3090/4090 (24GB) | $0 (owned) | ~50M params, solid |
| RunPod / Lambda Labs | ~$0.50/hr | ~350M params + flash attention |

Even the free Colab route works. Train for a few days, generate samples, document the learning curve. The story matters more than the final perplexity.

---

## Timeline (4 weeks)

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Tokenizer + dataset curation | tokenizer.py, processed dataset |
| 2 | Model implementation + training loop | model.py, train.py running |
| 3 | Full training run + evaluation | checkpoints, eval_report.md |
| 4 | Serving + README + deploy | api.py, app.py, Dockerfile, README |

---

## Summary

This capstone replaces the "Inference Observatory" and the vague "end-to-end GenAI product" capstone from month 6.

The project covers:
- **Tokenizer**: BPE from scratch
- **Architecture**: Decoder-only transformer (RoPE, GQA, RMSNorm, SwiGLU)
- **Training**: Full training loop with mixed precision, cosine schedule, gradient clipping
- **Evaluation**: Perplexity + generation benchmarks + baseline comparisons
- **Serving**: FastAPI + Gradio + Docker

After this, the person has a deployable model they built themselves, end-to-end. That clears most GenAI engineer interviews at the L4/L5 level for any company working with LLMs.
