# Adaptation and Fine-Tuning — Full Course Module

## Module Overview
This module builds a decision framework for choosing the right adaptation strategy, then dives deep into dataset design and parameter-efficient fine-tuning. Learners will understand when to tune vs prompt vs RAG, how to build high-quality datasets, and how to execute LoRA experiments with measurable evaluation.

## Target Audience
- Developers and technical professionals
- Comfortable with Python, APIs, and basic LLM concepts (Month 1–2 foundations)

## Learning Objectives
By the end of this module, learners will be able to:
- Decide when to use prompting, RAG, or fine-tuning for a given task
- Design and curate a high-quality labeled dataset with quality checks
- Run a parameter-efficient fine-tuning experiment using LoRA
- Evaluate a tuned model against a baseline with offline metrics
- Detect regression and overfitting in fine-tuned models

---

## Prerequisites
- Month 1: LLM architecture, tokenization, decoding strategies, prompting fundamentals
- Month 2: Tool use, agents, RAG pipeline concepts
- Python 3.10+
- Access to an LLM API (OpenAI, Anthropic, or local via Ollama)
- A machine with at least 8GB RAM for local tuning experiments (or access to a GPU)
- Basic familiarity with PyTorch or Hugging Face Transformers

---

## Module Structure

| Module | Topic | Lab |
|--------|-------|-----|
| 3.1 | When to Tune vs RAG | Compare prompt vs RAG for a domain task |
| 3.2 | Dataset Design | Create a small labeled dataset |
| 3.3 | Parameter-Efficient Tuning | Run a small LoRA experiment |
| Mini-Project | Domain-tuned model with measurable improvements | End-to-end system |

---

# Module 3.1: When to Tune vs RAG

## Core Concepts

### 1. The Adaptation Decision Framework

Not every problem needs fine-tuning. In fact, most problems should be solved with prompting or RAG first. Fine-tuning is the most expensive adaptation method — it requires data, compute, and careful evaluation. The key is knowing when it's worth the cost.

**The three adaptation axes:**

```
                    ┌─────────────────────┐
                    │   Adaptation Axis   │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Prompting     │  │   RAG           │  │  Fine-Tuning    │
│                 │  │                 │  │                 │
│ • Zero-shot     │  │ • Vector search │  │ • Full fine-tune│
│ • Few-shot      │  │ • Hybrid search │  │ • LoRA / QLoRA │
│ • Chain-of-     │  │ • Reranking     │  │ • Adapters      │
│   thought       │  │ • Citations     │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   Lowest cost           Medium cost            Highest cost
   Fastest iteration     Medium iteration       Slowest iteration
   Least control         Good control           Most control
```

**Decision tree for choosing an adaptation strategy:**

```
                           ┌─────────────────────┐
                           │  What is the task?  │
                           └─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       Needs to know         Needs to follow      Needs to change
       specific facts?       a specific          model behavior?
       (docs, DB,            format/structure?   (style, tone,
        real-time)                                domain knowledge)
              │                     │                     │
              ▼                     ▼                     ▼
         Use RAG               Use Prompting        Consider Tuning
         (cheapest,            (add few-shot       (only if prompt
          most updatable)       examples,           + RAG fail)
                                schema constraints)
```

**When to use each approach — detailed comparison:**

| Factor | Prompting | RAG | Fine-Tuning |
|--------|-----------|-----|-------------|
| **Cost to implement** | Free (API cost only) | Low (vector DB + API) | Medium-High (compute + data) |
| **Cost to update** | Instant (edit prompt) | Instant (re-index) | Days (re-train) |
| **Control over output** | Low | Medium | High |
| **Needs training data** | No (few-shot ok) | No | Yes (100s-1000s examples) |
| **Factual accuracy** | Low (hallucinates) | High (cites sources) | Medium (memorizes but can hallucinate) |
| **Knowledge cutoff** | Model's cutoff | Up to date | Model's cutoff + training data |
| **Model behavior change** | None | None | Significant |
| **Latency impact** | None | +search time | None |

### 2. The ROI of Fine-Tuning

Fine-tuning should only be considered when:

1. **Prompting fails consistently** — you've tried system prompts, few-shot examples, chain-of-thought, and output schemas, but the model still doesn't produce acceptable results.
2. **RAG is insufficient** — you need the model to internalize a domain, not just retrieve facts. Examples: medical diagnosis patterns, legal reasoning styles, code generation for a proprietary framework.
3. **You have high-quality data** — 500+ examples of (input → ideal output) pairs. More is better, but quality matters more than quantity.
4. **You can measure improvement** — you have an evaluation set, baseline metrics, and a clear target for what "better" means.

**The tuning decision checklist:**

```
Before tuning, ask yourself:
□ Have I tried 5+ prompt variants?
□ Have I tried few-shot with 5+ examples?
□ Have I tried RAG with a high-quality corpus?
□ Do I have 500+ labeled examples?
□ Do I have an evaluation set with 100+ examples?
□ Can I measure improvement with a clear metric?
□ Is the compute cost justified by the expected gain?

If you answered "No" to any of the first three → go back and try prompting/RAG first.
If you answered "No" to any of the last four → you are not ready to tune.
```

**Analogy:** Prompting is like giving instructions to a skilled contractor. RAG is like giving them a reference library. Fine-tuning is like sending them to a specialized training course. You wouldn't send someone to training if instructions and a library would suffice.

### 3. Real-World Case: When Tuning Makes Sense

**Scenario:** A medical coding system that maps clinical notes to ICD-10 codes.

| Approach | Result |
|----------|--------|
| Prompting | 60% accuracy — model guesses common codes but misses rare ones |
| Prompting + few-shot (10 examples) | 65% accuracy — slight improvement |
| RAG (coding guidelines) | 70% accuracy — better but misses nuanced mappings |
| Fine-tuning (2000 examples) | 88% accuracy — model learns coding patterns |

**Why tuning won here:** The task requires internalizing a complex coding ontology. RAG retrieval distracts with irrelevant guidelines. Prompting cannot convey the full mapping logic. The 2000 examples were carefully curated by medical coders.

---

## Lab 3.1: Compare Prompt vs RAG for a Domain Task

### Goal
Compare the effectiveness of prompting, few-shot prompting, and RAG for a domain-specific question-answering task.

### Steps
1. Choose a domain (e.g., medical, legal, finance, or your own field)
2. Prepare 5 domain-specific questions with ground-truth answers
3. Run each question with:
   - **Plain prompt**: "Answer this question: {question}"
   - **Few-shot prompt**: Include 3 example Q&A pairs before the question
   - **RAG prompt**: Retrieve 2 relevant context chunks, include in prompt with "Answer based only on the context"
4. Score each answer on accuracy (1-5 scale) and completeness (1-5 scale)
5. Record token usage and latency for each method

### Expected Observations
- Plain prompting may hallucinate or give generic answers
- Few-shot improves formatting and style but not factual accuracy
- RAG produces grounded answers but depends on retrieval quality
- No single method wins all cases — context matters

### Deliverable
A comparison table with accuracy, completeness, token cost, and latency for each method across all 5 questions.

---

## Exercises

1. **Decision Matrix**: Given these 4 scenarios, recommend prompt, RAG, or tune and justify:
   - A chatbot answering questions about your company's internal HR policies
   - A code generator that needs to output in your company's proprietary API style
   - A summarizer for recent news articles (today's date)
   - A creative writing assistant for fantasy world-building

2. **Prompt Engineering Audit**: Take a task where you think tuning is needed. Try 5 prompt variants first. Were any of them good enough? Document what you tried.

3. **Cost Projection**: Estimate the total cost (API calls, compute, human labeling time) for prompt/RAG vs tuning for a system serving 10K queries/month. Which is cheaper at what scale?

---

# Module 3.2: Dataset Design

## Core Concepts

### 1. What Makes a Good Tuning Dataset?

A fine-tuning dataset is a collection of (input, ideal output) pairs. The quality of this dataset is the single biggest factor in tuning success. A great model trained on bad data will produce bad results. A mediocre model trained on great data can excel.

**The data quality pyramid:**

```
                    ┌─────────────────────┐
                    │   Task Alignment    │  ← Does the data match the task?
                    ├─────────────────────┤
                    │   Consistency       │  ← Are examples labeled the same way?
                    ├─────────────────────┤
                    │   Coverage          │  ← Does the data cover all edge cases?
                    ├─────────────────────┤
                    │   Correctness       │  ← Are the labels accurate?
                    ├─────────────────────┤
                    │   Quantity          │  ← Do you have enough examples?
                    └─────────────────────┘
```

**Minimum dataset sizes (rule of thumb):**

| Task Type | Minimum Examples | Recommended | Notes |
|-----------|-----------------|-------------|-------|
| Classification | 100 per class | 500+ per class | Easier — smaller datasets work |
| Summarization | 200 | 1000+ | Harder — needs more diversity |
| Code generation | 300 | 2000+ | Very task-dependent |
| Instruction following | 500 | 5000+ | Broad behavior change |
| Chat / conversation | 500 pairs | 3000+ | Needs turn-taking diversity |

**Critical principle: Quality > Quantity**

A dataset with 500 carefully curated, diverse, correctly labeled examples will outperform a dataset with 5000 noisy, repetitive, inconsistent examples every time.

### 2. Dataset Design Process

**Step 1: Define the input-output contract**

Before collecting any data, define exactly what the model should receive and produce. This is called the **input-output contract** — think of it as an API specification for your dataset.

Every machine learning model learns patterns from examples. If those examples are inconsistent — same input, different outputs — the model learns nothing useful. The contract prevents this by forcing everyone (or every labeler) to agree on the format before labeling starts.

**Why this matters up front:**
- Two labelers given the same input will produce the same output structure
- The training code can parse every example the same way (no surprises)
- Evaluation can check outputs against the schema automatically
- You can spot missing fields before wasting time on bad data

**A good contract specifies:**
- **Input fields**: exactly what the model receives (name, type, allowed values)
- **Output fields**: exactly what the model should produce (structure, constraints, tone)
- **Rules**: edge cases, fallbacks, what to do when data is missing

```python
# Example: Customer support email response generator
INPUT_CONTRACT = {
    "fields": [
        "customer_email_text",   # The customer's incoming email
        "customer_tier",         # "bronze" | "silver" | "gold" | "platinum"
        "issue_category",        # "billing" | "technical" | "account" | "other"
        "previous_resolution_attempts",  # Free text or "none"
    ],
    "constraints": [
        "Email text may contain typos and informal language",
        "Tier must be one of the 4 defined values",
        "If no previous attempts, value is 'none'"
    ]
}

OUTPUT_CONTRACT = {
    "fields": [
        "response_email",        # The full response email text
        "action_taken",          # "refund" | "escalated" | "resolved" | "follow_up"
        "confidence",            # "high" | "medium" | "low"
    ],
    "constraints": [
        "Response must be professional and empathetic",
        "Action must be one of the 4 defined values",
        "Include specific reference to the customer's issue"
    ]
}
```

**Step 2: Choose a data collection strategy**

| Strategy | Cost | Quality | Speed | Best for |
|----------|------|---------|-------|----------|
| Human annotation | High | Highest | Slow | Production-grade datasets |
| LLM-generated | Low | Medium | Fast | Prototyping, augmentation |
| Existing logs | Low | Medium | Fast | If you have production data |
| Crowdsourcing | Medium | Variable | Medium | Large scale, simple tasks |
| Hybrid (LLM + human review) | Medium | High | Medium | Best balance for most teams |

**Step 3: Write labeling guidelines**

Labeling guidelines are the single most important tool for dataset quality. They ensure consistency across different labelers and over time.

```markdown
# Labeling Guidelines: Customer Support Response Dataset

## Task
Given a customer email + metadata, write the ideal support response.

## Rules
1. Always start with "Dear [Customer Name]," (infer from email if not given)
2. Acknowledge the specific issue in the first paragraph
3. Do not blame the customer for the issue
4. If a refund is requested, state the refund amount and timeline
5. Escalate if: security issue, legal threat, or 3+ follow-ups without resolution
6. End with "Best regards,\n[Support Team]"

## Quality Checklist (labelers must verify each)
□ Response addresses the customer's specific issue
□ Tone is professional and empathetic
□ Action taken matches the response content
□ No hallucinated information (policies, refunds, etc.)
□ Grammar and spelling are correct
□ Response length is 3-7 sentences
```

**Step 4: Collect and review in rounds**

```
Round 1: Label 50 examples → Review → Revise guidelines
Round 2: Label 100 examples → Review → Check inter-labeler agreement
Round 3: Label remaining examples → Final review → Split train/val/test
```

Once you have multiple labelers (or even just yourself on different days), you must measure **inter-labeler agreement** — how often do they pick the same label for the same input?

**The math is simple**: count matches, divide by total. If two labelers agree 70% of the time, your guidelines aren't specific enough. The industry target is **90%+**.

Low agreement means the task definition is ambiguous. Fix your guidelines, not the labelers. Run a small batch (50 examples), check agreement, revise guidelines, repeat. This iterative process is the single highest-leverage quality investment you can make.

```python
def check_agreement(labeler_a: list, labeler_b: list) -> float:
    """Calculate simple agreement rate between two labelers."""
    matches = sum(1 for a, b in zip(labeler_a, labeler_b) if a == b)
    return matches / len(labeler_a)

# Example: Check action_taken field agreement
a_actions = ["refund", "escalated", "resolved", "refund", "follow_up"]
b_actions = ["refund", "escalated", "resolved", "refund", "resolved"]
agreement = check_agreement(a_actions, b_actions)
print(f"Agreement rate: {agreement:.0%}")  # 80% — target is 90%+
```

### 3. Data Quality Assurance

**Common data quality issues and how to catch them:**

| Issue | Symptom | Detection Method |
|-------|---------|-----------------|
| Label noise | Model learns wrong patterns | Spot-check 10% of labels, measure labeler agreement |
| Duplicates | Model overfits to repeated examples | Hash and deduplicate inputs |
| Distribution shift | Train accuracy high, eval low | Compare train vs eval distributions |
| Missing edge cases | Model fails on real-world inputs | Coverage analysis against a taxonomy |
| Label bias | Model produces biased outputs | Audit labels for demographic fairness |
| Format inconsistency | Model outputs wrong format | Validate all outputs against schema |

**Data versioning — treat your data like code:**

A dataset is a living artifact. You'll fix incorrect labels, add new examples, remove duplicates, reformat fields. Without versioning, you'll lose track of what changed and why.

A **dataset card** solves this. It's a metadata header embedded with every dataset version that records:

- **Identity**: version number, creation date, what changed since last version
- **Provenance**: who labeled it, which guidelines they followed, where the raw data came from
- **Statistics**: how many examples in train/val/test, class balance, average length
- **Integrity**: checksums (hashes) of every split file — if a file changes, its hash changes, and you know it's been tampered with

Treat your dataset card like a `package.json` or `requirements.txt` — it's the first thing someone checks to understand what they're working with.

```python
# Dataset card — metadata for every dataset version
dataset_card = {
    "version": "1.0.3",
    "created": "2025-06-11",
    "source": "human_labeling",
    "num_examples": 1500,
    "train/val/test": "1200/150/150",
    "labelers": ["alice@co", "bob@co", "carol@co"],
    "avg_labeler_agreement": 0.93,
    "fields": ["customer_email", "tier", "category", "prior_attempts"],
    "output_fields": ["response_email", "action_taken", "confidence"],
    "guidelines_version": "2.1",
    "checksums": {
        "train": "a1b2c3d4...",
        "val": "e5f6g7h8...",
        "test": "i9j0k1l2..."
    }
}
```

**Dataset format — JSONL is standard:**

JSONL (JSON Lines) is the standard interchange format for fine-tuning datasets. Every major training framework (Hugging Face, OpenAI, Axolotl, Unsloth) supports it.

**Why JSONL and not CSV or plain JSON?**

- **Streamable**: Each line is a self-contained record. You can process a million examples by reading one line at a time — no need to load the entire dataset into RAM.
- **Schema-flexible**: Different examples can have different fields. A CSV requires every row to have the same columns; JSONL lets each example have its own structure.
- **Version-control friendly**: Each line is one example, so diffs between dataset versions show exactly which examples changed. With a giant JSON array, a single change shifts every line after it.
- **Universal**: Every language has a JSON parser. No special tooling needed.

**Standard format for instruction tuning:**

Each line has an `"input"` field (what the model receives) and an `"output"` field (what it should produce). Some frameworks use `"instruction"` and `"response"` instead — check the specific trainer's expected schema.

```jsonl
{"input": {"text": "What is the refund policy for late deliveries?"}, "output": "Our refund policy states that if your order arrives more than 5 business days late, you are eligible for a full refund. Please contact support with your order number to initiate the process."}
{"input": {"text": "How do I reset my password?"}, "output": "To reset your password, go to the login page and click 'Forgot Password'. You will receive a reset link via email. If you do not see the email within 5 minutes, check your spam folder."}
```

---

## Lab 3.2: Create a Small Labeled Dataset

### Goal
Build a high-quality labeled dataset of at least 50 (input → output) pairs for a domain task of your choice.

### Steps
1. Define your input-output contract (schema + constraints)
2. Write labeling guidelines (at least 5 specific rules)
3. Collect 50 raw inputs (from real data if possible, or write realistic examples)
4. Label all 50 examples yourself following your guidelines
5. Review your own labels — identify 3 examples where your labeling was inconsistent
6. Revise your guidelines based on review
7. Split data: 35 train, 10 val, 5 test
8. Write a dataset card with version, stats, and checksums

### Expected Observations
- Labeling is harder and slower than expected (aim for 10-20 examples/hour)
- Inconsistencies appear around edge cases — guidelines need constant refinement
- A small high-quality dataset is more valuable than a large noisy one

### Deliverable
A dataset card plus JSONL files for train/val/test splits.

---

## Exercises

1. **Guideline Audit**: Given this labeling guideline — "Write a polite response" — explain why it is insufficient. Write a better version with at least 5 specific, checkable rules.

2. **Edge Case Identification**: For a sentiment classification task (positive/negative/neutral), list 10 edge cases where the label is ambiguous. How would your guidelines handle each?

3. **Data Augmentation**: Take 5 examples from your dataset and use an LLM to generate 3 variants of each (rephrase input, keep output the same). Check if the variants are realistic and useful.

---

# Module 3.3: Parameter-Efficient Tuning

## Core Concepts

### 1. What is Parameter-Efficient Fine-Tuning (PEFT)?

**The core problem:**

Every time you want to adapt a model to a new task, you have two options:
1. **Full fine-tuning**: Update all 7B+ parameters. This works but is brutally expensive — you need multiple GPUs, days of training, and you end up with a 30GB model file for every variant.
2. **Don't tune at all**: Just use prompts. This is cheap but limited — the model can't learn new patterns it doesn't already know.

PEFT offers a third path: **update almost nothing, get most of the benefit**.

**The key insight:**

A model's weights are highly redundant. A 4096×4096 weight matrix doesn't have 16.8M independent degrees of freedom — it lives on a much lower-dimensional manifold. LoRA exploits this by learning a tiny set of parameters that sit in *addition* to the frozen weights, steering the output without altering what the model already knows.

**Analogy:** Think of the base model as a highly skilled musician who already knows how to play any instrument. Full fine-tuning is like retraining them from scratch for a new genre. PEFT is like giving them a small set of sheet music (the adapter) — the musician's core skills stay intact, but the output changes to match the new piece.

Full fine-tuning updates all model weights — billions of parameters. It requires massive compute, multiple GPUs, days of training, and produces a full copy of the model for each variant. PEFT methods update only a small fraction of parameters while keeping the rest frozen.

**Why PEFT matters:**

```
Full Fine-Tuning:
┌─────────────────────────────────────────────┐
│  Update ALL 7B parameters                   │
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
│  Needs: 28GB model × optimizer states       │
│  ≈ 56-112GB VRAM total                      │
│  Produces: full 7B model copy per variant   │
└─────────────────────────────────────────────┘

LoRA (Low-Rank Adaptation):
┌─────────────────────────────────────────────┐
│  Freeze original 7B weights                 │
│  Insert tiny trainable adapters:            │
│  ■ ■ ■ (0.1-1% of total params)            │
│  Needs: 28GB base model + ~300MB adapters   │
│  Produces: tiny adapter file per variant    │
└─────────────────────────────────────────────┘
```

**The core idea of LoRA:**

Instead of modifying the weight matrix W directly (which has shape d × k), LoRA learns a low-rank decomposition:

```
W' = W + ΔW    where ΔW = A × B

W:      original frozen weights  (d × k)
A:      learned matrix           (d × r)  — initialized with random values
B:      learned matrix           (r × k)  — initialized with zeros
r:      rank (typically 4-64)    — small number
ΔW:     low-rank update          (d × k)  — but stored as A×B (d×r + r×k parameters)
```

The rank `r` controls the expressiveness vs efficiency tradeoff:
- r=4: Very efficient, captures simple patterns (good for classification)
- r=16: Balanced (good for most tasks)
- r=64: More expressive, captures complex patterns (good for creative tasks)

**Parameter count comparison:**

```
For a 7B model with attention weight d=4096, k=4096:

Full fine-tune:   4096 × 4096 = 16.8M params per layer
LoRA r=8:         4096×8 + 8×4096 = 65,536 params per layer (0.39% of full)
LoRA r=64:        4096×64 + 64×4096 = 524,288 params per layer (3.1% of full)

Total for full fine-tune: ~7B parameters
Total for LoRA r=8:      ~33M parameters (0.47% of full)
```

### 2. LoRA Configuration and Hyperparameters

**Understanding the knobs you can turn:**

LoRA gives you several configuration options that control the tradeoff between adaptation strength, compute cost, and overfitting risk. These are not arbitrary — each one maps to a specific aspect of how the adapter works:

- **Rank (r)**: How many dimensions the adapter has. Higher r = more expressive = can learn more complex patterns but also more likely to overfit. Think of rank as the "width" of the bottleneck between A and B.
- **Alpha (α)**: How strongly the adapter's output scales before being added to the base model's output. Higher alpha = stronger adaptation effect.
- **Target modules**: Which specific weight matrices in the transformer get adapters. Attention layers (Q, K, V, O) are the most impactful; feed-forward layers matter less for most tasks.
- **Dropout**: Randomly drops adapter neurons during training to prevent co-adaptation. Higher dropout = more regularization = less overfitting.

**The alpha/r ratio matters more than either value alone:**

The formula `output = base + (α/r) × adapter_output` means α and r are coupled. If you double both, the ratio stays the same and the effective adaptation strength is unchanged. A ratio around 2 (e.g. α=16, r=8) is a safe starting point for most tasks.

**Key LoRA hyperparameters:**

| Hyperparameter | What it controls | Typical values | Effect |
|---------------|-----------------|----------------|--------|
| `r` (rank) | Expressiveness of adaptation | 4, 8, 16, 32, 64 | Higher = more capacity, more overfitting risk |
| `alpha` (scaling) | Strength of LoRA update | 8, 16, 32 | Higher = stronger adaptation |
| `target_modules` | Which layers to adapt | `["q_proj", "v_proj"]` or all | More modules = more adaptation |
| `dropout` | Regularization | 0.05, 0.1 | Higher = less overfitting |
| `bias` | Whether to train biases | "none", "all", "lora_only" | Usually "none" |

**How alpha and r interact:**

```
LoRA output = base_output + (alpha/r) * lora_output
```

The `alpha/r` ratio controls the update strength. A common starting point:
- `alpha = 16`, `r = 8` → ratio = 2.0
- `alpha = 32`, `r = 16` → ratio = 2.0
- `alpha = 16`, `r = 16` → ratio = 1.0

If your tuned model isn't changing enough (underfitting), try increasing α or decreasing r (raising the ratio). If it's changing too much or overfitting, decrease α or increase r (lowering the ratio).

**Where to apply LoRA — which layers matter most:**

A transformer layer has two main sub-components: **self-attention** (where tokens communicate with each other) and **feed-forward** (where each token processes information independently). Within self-attention, there are four weight matrices:

- **W_q (Query)**: Decides "what am I looking for?" — projects each token into a query vector
- **W_k (Key)**: Decides "what do I contain?" — projects each token into a key vector
- **W_v (Value)**: Decides "what information do I carry?" — projects each token into a value vector
- **W_o (Output)**: Combines the attention results back into the original dimension

**Starting with q_proj and v_proj** is the standard recommendation because:
- Q and V have the most influence over attention patterns (what the model pays attention to)
- K changes are riskier (can break the attention distribution)
- O and feed-forward changes add capacity but increase overfitting risk

The pseudo-code diagram below shows where each matrix sits in the transformer layer:

```
┌─────────────────────────────────────────┐
│  Transformer Layer                      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  Self-Attention                 │    │
│  │  Q: W_q (apply LoRA here ✓)    │    │
│  │  K: W_k (optional)             │    │
│  │  V: W_v (apply LoRA here ✓)    │    │
│  │  O: W_o (optional)             │    │
│  └─────────────────────────────────┘    │
│         ↓                               │
│  ┌─────────────────────────────────┐    │
│  │  Feed-Forward                   │    │
│  │  W_up (optional)                │    │
│  │  W_down (optional)              │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

Best practice: Start with q_proj + v_proj. If results are poor, add k_proj, o_proj, then feed-forward layers.
```

### 3. The Full Fine-Tuning Workflow

```
                    ┌─────────────────────┐
                    │   Base Model        │
                    │   (pre-trained)     │
                    └─────────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │   Load Tokenizer    │
                    │   + Format Data     │
                    └─────────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │   Apply LoRA Config │
                    │   (freeze base,     │
                    │    init adapters)   │
                    └─────────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │   Training Loop     │
                    │   • Forward pass    │
                    │   • Compute loss    │
                    │   • Backprop (only  │
                    │     LoRA params)    │
                    │   • Optimizer step  │
                    └─────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
            ┌─────────────────┐ ┌─────────────────┐
            │ Save LoRA       │ │ Evaluate on     │
            │ adapter weights │ │ validation set  │
            └─────────────────┘ └─────────────────┘
```

**Complete LoRA training script — step-by-step walkthrough:**

Before reading the code, understand the overall flow:

```
Load pre-trained model (frozen)
           │
   Apply LoRA adapters (trainable)
           │
   Format data as instruction text
           │
   Tokenize text → number sequences
           │
   Training loop:
     ├─ Forward pass: tokens → predictions
     ├─ Compute loss: predictions vs targets
     ├─ Backprop: gradients flow ONLY through LoRA adapters
     └─ Optimizer step: update LoRA weights
           │
   Save LoRA adapter (tiny file, ~MB)
```

The script below does exactly this. Here's what each section does:

1. **Load model & tokenizer** — We use Phi-2 (2.7B params), a small model that fits on most GPUs. The tokenizer converts text to token IDs (integers). We set `pad_token = eos_token` because many models don't have a native pad token, and batching requires uniform sequence lengths. `device_map="auto"` tells Transformers to spread the model across available GPUs/CPU automatically.

2. **Apply LoRA** — We create a `LoraConfig` describing which layers to adapt (`q_proj`, `v_proj` — the query and value weight matrices in attention). The base model's weights stay frozen. `get_peft_model` wraps the base model with the LoRA adapters — only ~0.3% of parameters are now trainable. The print confirms: out of 2.8B total parameters, only 8.4M will be updated.

3. **Prepare dataset** — Raw examples (input/output pairs) are formatted into a single text string: `"Instruction: {input}\nResponse: {output}"`. This matches how the model was pre-trained (predicting the next token). The Hugging Face `Dataset` is then tokenized: each text is truncated/padded to 512 tokens, producing `input_ids` (the token sequence) and `labels` (same as input_ids for language modeling — the model predicts each next token).

4. **Training arguments** — Configuration for the Hugging Face `Trainer`:
   - `per_device_train_batch_size=4`: 4 examples per GPU per step
   - `gradient_accumulation_steps=4`: accumulate gradients over 4 steps before updating → effective batch size = 4 × 4 = 16
   - `num_train_epochs=3`: 3 full passes over the dataset
   - `learning_rate=2e-4`: LoRA needs higher LR than full fine-tuning (typical range: 1e-4 to 3e-4)
   - `fp16=True`: mixed precision training — faster and uses less VRAM
   - `save_strategy="epoch"`: save a checkpoint after each epoch
   - `report_to="none"`: no experiment tracking (WandB/TensorBoard)

5. **Train** — The Trainer handles everything: batching, forward pass, loss computation, backpropagation, optimizer step, logging, checkpointing. Because the base model is frozen, only the LoRA adapter parameters receive gradients.

6. **Save** — Only the LoRA adapter weights and the tokenizer are saved. The file is a few MB — trivial to share, version, or deploy alongside the base model.

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# 1. Load base model and tokenizer
model_name = "microsoft/phi-2"  # Small model for local prototyping
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Apply to query and value projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 2,789,000,000 || trainable%: 0.30

# 3. Prepare dataset
train_data = [
    {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"input": "Explain gravity simply.", "output": "Gravity is a force that pulls objects with mass toward each other. It's why apples fall from trees and why we stay on the ground."},
    # ... more examples
]

def format_example(example):
    """Format as instruction-style text."""
    return {
        "text": f"Instruction: {example['input']}\nResponse: {example['output']}"
    }

formatted_data = [format_example(ex) for ex in train_data]
dataset = Dataset.from_list(formatted_data)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./lora-adapters",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",  # Disable wandb/tensorboard for local runs
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# 6. Save the LoRA adapter (small file, ~MBs)
model.save_pretrained("./lora-adapters/final")
tokenizer.save_pretrained("./lora-adapters/final")

print("Training complete! Adapter saved to ./lora-adapters/final")
```

### 4. Loading and Using a LoRA Adapter

After training, you have **two separate artifacts**:

1. **The base model** — unchanged, frozen, several GB. You keep one copy.
2. **The LoRA adapter** — tiny file (~MB) containing just the trained ΔW = A × B weights.

This separation is the key advantage of LoRA. You can:

- **Train multiple adapters** for different tasks, each just a few MB
- **Swap adapters at inference time** without reloading the base model
- **Share adapters** — upload a 5MB file instead of a 30GB model
- **Roll back** by simply loading a different adapter or none at all

Loading works in two steps: load the base model as usual, then attach the adapter on top with `PeftModel.from_pretrained`. The adapter weights are added to the base model's forward pass automatically — no manual merging needed.

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "./lora-adapters/final")

# Inference — use exactly like the base model
tokenizer = AutoTokenizer.from_pretrained("./lora-adapters/final")
inputs = tokenizer("Instruction: What is machine learning?\nResponse:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 5. Evaluating a Tuned Model — Regression Testing

The biggest risk in fine-tuning is **regression** — the model gets better at the target task but worse at everything else. You must test for this.

**Why separate evaluation is necessary:**

When you train a model, the loss function only measures performance on your training data. It has no idea whether the model still knows basic facts, follows safety rules, or formats output correctly. A model can score 95% on your custom dataset while dropping from 85% to 60% on general knowledge — and you'd never know unless you test for it.

**What you need to measure:**

| Test | Purpose | What to look for |
|------|---------|-----------------|
| Target task (held-out) | Did we improve? | Tuned > Base by meaningful margin |
| General QA | Did we lose knowledge? | Tuned ≈ Base (within 5%) |
| Safety/refusal | Did we break guardrails? | Tuned ≈ Base on refusal rate |
| Format compliance | Did output structure change? | Tuned matches expected schema |
| Output length | Did verbosity shift? | Similar token count |

A drop of more than 5% on any general capability is a red flag. You may need to reduce LoRA rank, increase dropout, or mix general-domain data into your training set.

**The evaluation framework below:**

The `TuningEvaluator` class wraps both the base and tuned models. It runs the same inputs through both, computes a score (accuracy, F1, ROUGE, etc.), and reports the difference (Δ). Positive Δ = improvement. Negative Δ = regression.

The `regression_check` method specifically tests multiple general-purpose eval sets and flags any where the tuned model scores >5% worse than the base. This catches regression automatically so you don't have to remember to check each metric manually.

```python
from sklearn.metrics import accuracy_score, f1_score

class TuningEvaluator:
    def __init__(self, base_model, tuned_model, tokenizer):
        self.base_model = base_model
        self.tuned_model = tuned_model
        self.tokenizer = tokenizer

    def evaluate_task(self, eval_dataset, task_name):
        """Evaluate both models on a specific task."""
        base_scores = self._run_eval(self.base_model, eval_dataset)
        tuned_scores = self._run_eval(self.tuned_model, eval_dataset)

        print(f"\n=== {task_name} ===")
        print(f"  Base model:  {base_scores}")
        print(f"  Tuned model: {tuned_scores}")
        print(f"  Δ: {tuned_scores - base_scores:+.2f}")

        return {"base": base_scores, "tuned": tuned_scores}

    def regression_check(self, general_eval_sets):
        """Check if tuning degraded general capabilities."""
        print("\n=== Regression Check ===")
        regressions = []
        for eval_set in general_eval_sets:
            result = self.evaluate_task(eval_set["data"], eval_set["name"])
            if result["tuned"] < result["base"] - 0.05:
                regressions.append(eval_set["name"])

        if regressions:
            print(f"\n⚠️  Regression detected in: {', '.join(regressions)}")
        else:
            print("\n✓ No significant regression detected")
        return regressions
```

**What to evaluate:**

| Evaluation Set | Purpose | Metric |
|---------------|---------|--------|
| Target task (held-out) | Did we improve? | Accuracy, F1, ROUGE |
| General QA | Did we lose general knowledge? | Accuracy |
| Safety / refusal | Did we break guardrails? | Pass rate |
| Format adherence | Did we mess up output format? | Schema compliance |
| Latency | Did we slow down? | Tokens/sec |
| Output length | Did verbosity change? | Avg tokens per response |

### 6. QLoRA — Quantized LoRA for Consumer Hardware

QLoRA quantizes the base model to 4-bit before training, drastically reducing memory requirements. This makes fine-tuning 7B+ models possible on a single consumer GPU (or even CPU+RAM for smaller models).

```
Full fine-tune 7B:    56-112GB VRAM (2-4 A100s)
LoRA 7B:              14-28GB VRAM (1 A100 or 2 RTX 3090s)
QLoRA 7B (4-bit):     4-8GB VRAM (1 RTX 3090 or even RTX 4080)
```

**How QLoRA works under the hood:**

QLoRA compresses the base model weights from 16-bit floats to 4-bit integers using a technique called **NF4 (Normal Float 4)**. Unlike naive 4-bit quantization that uniformly distributes values, NF4 adjusts the quantization levels to match the bell-curve distribution of neural network weights — most weights cluster near zero, so more precision is allocated there.

**Double quantization** (the `bnb_4bit_use_double_quant=True` flag) then quantizes the quantization constants themselves, saving another ~0.5GB. The result is near-lossless compression for training purposes — models fine-tuned with QLoRA match FP16 LoRA quality within <1%.

**What changes vs regular LoRA:**

- **Loading**: You pass a `BitsAndBytesConfig` to `from_pretrained` instead of just `torch_dtype`. The model loads in 4-bit automatically.
- **Storage**: Base weights stay 4-bit throughout training. They're dequantized to FP16 on-the-fly during the forward pass, then quantized back — this is the ~10% speed penalty.
- **Training**: Everything else is identical — same `LoraConfig`, same `Trainer`, same saving/loading of adapters. The LoRA adapters themselves are still trained in full FP16 precision.

**Memory breakdown for a 7B model:**

| Component | LoRA (FP16) | QLoRA (4-bit) |
|-----------|-------------|---------------|
| Base model weights | 28 GB | ~3.5 GB |
| LoRA adapters | ~50 MB | ~50 MB |
| Optimizer states (Adam) | ~2× adapter size | ~2× adapter size |
| Activations + gradients | varies by batch size | varies by batch size |
| **Typical total** | **28-32 GB** | **4-8 GB** |

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load 4-bit base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Apply LoRA on top of quantized model
model = get_peft_model(model, lora_config)
# Train normally — memory usage is ~5-8GB for a 7B model
```

---

## Lab 3.3: Run a Small LoRA Experiment

### Goal
Fine-tune a small language model on a domain task using LoRA and measure improvement over the base model.

### Steps
1. Choose a base model small enough to run locally: `microsoft/phi-2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, or `google/gemma-2b`
2. Prepare a dataset of 50-100 examples (use the dataset from Lab 3.2 if available)
3. Configure LoRA with r=8, alpha=16 on q_proj and v_proj
4. Train for 3 epochs with learning rate 2e-4
5. Evaluate the tuned model vs base model on:
   - A held-out test set (10 examples)
   - 3 general knowledge questions (to check regression)
   - Latency comparison
6. Document: What improved? What regressed? Was the tuning worth it?

### Expected Observations
- The tuned model should perform better on the target task
- Small regressions on general knowledge may occur
- LoRA training should complete in minutes for small models
- Overfitting may occur if the dataset is too small (<50 examples)

### Deliverable
A Python script that trains and evaluates a LoRA model, plus a report documenting the comparison.

---

## Exercises

1. **Hyperparameter Sweep**: Train LoRA with r=4, r=8, and r=16 on the same dataset. Compare target task accuracy and general knowledge retention. Which rank gives the best tradeoff?

2. **Regression Detection**: Given a tuned model that scores 92% on the target task (up from 60%) but drops from 85% to 72% on general QA, what do you recommend? How would you fix the regression?

3. **QLoRA vs LoRA**: Compare training memory usage, training time, and output quality between LoRA (FP16) and QLoRA (4-bit) for the same model and dataset.

---

## Mini-Project: Domain-Tuned Model with Measurable Improvements

### Goal
Build a complete fine-tuning pipeline from dataset creation to evaluation, producing a domain-tuned model with demonstrable gains over the base model.

### Requirements
1. **Data**: A labeled dataset of at least 200 examples in a domain of your choice
2. **Training**: LoRA or QLoRA fine-tuning on a base model
3. **Evaluation**: At least 3 metrics (target task accuracy, general QA, format compliance)
4. **Comparison**: Baseline (base model) vs tuned model on the same evaluation sets
5. **Regression check**: Verify that general capabilities are not degraded beyond an acceptable threshold
6. **Reproducibility**: All code, data, and config in a single directory

### Suggested Domains
- **Customer support**: Emails → response + action
- **Medical advice QA**: Symptom → explanation + recommendation
- **Code generation**: Natural language → SQL queries
- **Legal document parsing**: Clause → plain language summary
- **Financial analysis**: Earnings report → key metrics

### Deliverable
A GitHub-ready repository containing:
- `train.py` — training script with configurable hyperparameters
- `evaluate.py` — evaluation script with baseline comparison
- `data/` — train/val/test JSONL files
- `adapters/` — saved LoRA adapter weights
- `report.md` — results summary with metrics table

### Rubric (100 points)
- **Dataset quality (25 points)**: Clear schema, consistent labeling, proper train/val/test split, dataset card
- **Training correctness (25 points)**: LoRA configuration, training loop, checkpointing
- **Evaluation rigor (25 points)**: Baseline comparison, multiple metrics, regression check
- **Results and documentation (25 points)**: Clear metrics table, analysis of what improved/regressed, reproducibility

---

## Assessment: Quick Quiz (5 Questions)

1. **When should you choose fine-tuning over RAG?**
   Fine-tuning is appropriate when: (a) prompting and RAG have been tried and consistently fail, (b) you need the model to internalize domain patterns rather than retrieve facts, (c) you have 500+ high-quality labeled examples, and (d) you can measure improvement with clear metrics. Tuning is overkill if RAG or prompting suffice.

2. **What is the minimum recommended dataset size for a classification fine-tuning task?**
   100 examples per class as a minimum, with 500+ per class recommended. Quality (consistency, coverage, correctness) matters more than quantity — 500 excellent examples beat 5000 noisy ones.

3. **How does LoRA reduce memory requirements compared to full fine-tuning?**
   LoRA freezes all original model weights and inserts small low-rank adapter matrices (A and B) at specific layers. Only the adapter parameters are trained, reducing trainable parameters from billions to millions (0.1-1% of total). This cuts optimizer state memory by ~90% and adapter storage to MBs instead of GBs.

4. **What is regression in the context of fine-tuning, and how do you detect it?**
   Regression is when the tuned model performs worse on general capabilities it previously handled well. Detection requires evaluating the tuned model on a diverse set of general tasks (QA, safety, format adherence) and comparing scores to the base model. A drop of >5% on any general task should trigger investigation.

5. **What three things should be in a dataset card?**
   (1) Version metadata (version number, creation date, source, checksums), (2) Statistics (num examples, train/val/test split, class distribution), (3) Quality metrics (labeler agreement, guidelines version, known issues). The dataset card ensures reproducibility and helps others understand the data's limitations.

---

## Common Pitfalls and How to Address Them

- **Tuning before trying prompt/RAG** — Fine-tuning is expensive and irreversible. Always exhaust prompt engineering and RAG first. *Solution*: Use the decision checklist before starting any tuning project.

- **Using noisy or inconsistent labels** — The model learns exactly what you give it. Inconsistent labels confuse the model. *Solution*: Write detailed labeling guidelines, measure inter-labeler agreement, and spot-check labels regularly. Iterate on guidelines in rounds.

- **Overfitting to a small dataset** — With fewer than 200 examples, the model may memorize rather than generalize. *Solution*: Use higher LoRA rank for more capacity, add dropout, use early stopping, and evaluate on a held-out set after every epoch.

- **Forgetting to check regression** — The tuned model may solve the target task brilliantly but fail at basic things the base model handled fine. *Solution*: Always evaluate on a diverse general-knowledge test set before and after tuning. If regression is detected, reduce LoRA rank, increase dropout, or mix general data into the training set.

- **Applying LoRA to too few or too many modules** — Only adapting q_proj may be insufficient; adapting every module may overfit. *Solution*: Start with q_proj + v_proj. If underfitting, add k_proj and o_proj. If overfitting, remove modules or increase dropout.

- **Using too high a learning rate** — LoRA is sensitive to learning rate. Standard fine-tuning rates (5e-5) are often too high. *Solution*: Use 1e-4 to 3e-4 for LoRA. Monitor loss during training and reduce LR if loss spikes or oscillates.

---

## Resources

- **Papers**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021), "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)
- **Libraries**: Hugging Face PEFT, Transformers, Datasets, BitsAndBytes
- **Tools**: Weights & Biases for experiment tracking, Hugging Face Hub for model sharing
- **Datasets**: Hugging Face Datasets hub for inspiration and reference datasets
- **Guides**: PEFT documentation, Hugging Face NLP course (fine-tuning chapter)
