# BERT Attention Weights — Study Guide

A practical walkthrough of how to extract and interpret attention weights from BERT using HuggingFace Transformers.

---

## What is attention?

BERT is a transformer model. At its core, every transformer layer runs a mechanism called **multi-head self-attention**. For every token in your sentence, attention computes a score against every other token — "how relevant is token B when I'm trying to represent token A?" Those scores are normalised with softmax (so they sum to 1.0) and used to build each token's new representation.

BERT has:
- **12 layers** — stacked one on top of the other
- **12 attention heads per layer** — each head learns to specialise in a different pattern (syntax, coreference, positional distance, etc.)
- **1 attention matrix per head** — shape `(seq_len, seq_len)`

So for a single sentence you get `12 × 12 = 144` attention matrices to explore.

---

## Setup

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
```

> **Note:** You must load the model with `attn_implementation="eager"` if you want `output_attentions=True` to work. The default backend (`sdpa`) fuses the computation and cannot return intermediate weight matrices.

---

## Step 1 — Load the model and tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModel.from_pretrained(
    "bert-base-uncased",
    attn_implementation="eager"   # required for output_attentions
)
```

### What is AutoTokenizer?

`AutoTokenizer` is a smart loader — it reads the model card and picks the right tokenizer class automatically. For `bert-base-uncased` it uses **WordPiece** tokenization, which splits words into sub-word pieces:

- `"tired"` → stays as `["tired"]`
- `"unbelievable"` → becomes `["un", "##believable"]`

The `##` prefix means *"this piece continues the previous word."* This matters when you're indexing into the token list — your sentence may have more tokens than words.

### What is AutoModel?

`AutoModel` loads the base transformer (no task-specific head on top). It returns raw hidden states and, when asked, attention weights. If you wanted classification you'd use `AutoModelForSequenceClassification`, etc.

---

## Step 2 — Tokenize your sentence

```python
sentence = "The cat sat on the mat because it was tired"

inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print(tokens)
# ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '[SEP]']
```

### What is `return_tensors="pt"`?

Tells the tokenizer to return PyTorch tensors instead of plain Python lists. The model expects tensors as input.

### What are `[CLS]` and `[SEP]`?

BERT always wraps your input in two special tokens:

| Token | Purpose |
|-------|---------|
| `[CLS]` | Classification token. Sits at position 0. Its final hidden state is used as a sentence-level representation for classification tasks. |
| `[SEP]` | Separator token. Marks the end of a segment (or the boundary between two sentences in next-sentence tasks). |

These are real positions in the attention matrix — you'll see them when you plot the heatmap.

### What does `convert_ids_to_tokens` do?

The tokenizer works with integer IDs internally (e.g. `the` → `1996`). `convert_ids_to_tokens` maps each ID back to its readable string — you need this to label the axes on your heatmap.

---

## Step 3 — Run the forward pass

```python
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions

print(f"Layers: {len(attentions)}")          # 12
print(f"Shape per layer: {attentions[0].shape}")  # torch.Size([1, 12, 12, 12])
```

### Why `torch.no_grad()`?

During training, PyTorch tracks every operation in a computation graph so it can calculate gradients for backprop. During inference (i.e. just running the model to get output) you don't need any of that. `torch.no_grad()` disables gradient tracking — it saves memory and makes the forward pass faster.

### Why `output_attentions=True`?

By default BERT computes the attention weights internally and discards them — they're only needed to produce the hidden states. Passing `output_attentions=True` tells the model to keep them and return them in the output object.

### What is `outputs.attentions`?

A **tuple of 12 tensors**, one per layer. Each tensor has shape:

```
(batch_size, num_heads, seq_len, seq_len)
   = (1,        12,       12,      12)
```

- **batch_size** — how many sentences you passed in (1 in our case)
- **num_heads** — 12 attention heads
- **seq_len × seq_len** — the full attention matrix: `matrix[i][j]` = how much token `i` attends to token `j`

Every row sums to 1.0 (softmax is applied across each row).

---

## Step 4 — Visualise a single head (heatmap)

```python
layer, head = 0, 0
attn_matrix = attentions[layer][0, head].numpy()  # shape: (seq_len, seq_len)

plt.figure(figsize=(10, 8))
plt.imshow(attn_matrix, cmap="Blues")
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.colorbar(label="Attention weight")
plt.title(f"Attention heatmap — Layer {layer}, Head {head}")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()
```

### Breaking down the indexing

```python
attentions[layer]          # picks layer 0  → shape (1, 12, 12, 12)
attentions[layer][0]       # picks batch 0  → shape (12, 12, 12)
attentions[layer][0, head] # picks head 0   → shape (12, 12)
```

`.numpy()` converts the PyTorch tensor to a NumPy array so matplotlib can plot it.

### Reading the heatmap

- **Rows** = the "query" token (the one doing the attending)
- **Columns** = the "key" token (the one being attended to)
- **Colour** = attention weight — darker blue means the query token is paying more attention to that key token

In early layers (layer 0) you'll typically see strong diagonal patterns — tokens attending heavily to themselves and their immediate neighbours. Semantic patterns like coreference emerge in middle layers (4–8).

### Use different heads

```python
# Try different heads to see different specialisations
for head in range(12):
    attn_matrix = attentions[0][0, head].numpy()
    # plot ...
```

Each head has learned something different. Some common patterns researchers have found:
- **Heads that track [SEP]** — attend from most tokens to the separator (a form of "no-op")
- **Syntactic heads** — follow dependency relations (subject → verb, adjective → noun)
- **Coreference heads** — pronouns attending to their antecedents

---

## Step 5 — Average attention across all heads

```python
avg_attn = attentions[layer][0].mean(dim=0).numpy()

print("\nAverage attention from [CLS] to each token (layer 0):")
for tok, score in zip(tokens, avg_attn[0]):
    print(f"  {tok:15s} → {score:.4f}")
```

### What does `.mean(dim=0)` do?

`attentions[layer][0]` has shape `(12, seq_len, seq_len)` — 12 heads stacked. `.mean(dim=0)` collapses the head dimension by averaging, giving a single `(seq_len, seq_len)` matrix representing the layer's collective attention.

### Why look at row 0 (`avg_attn[0]`)?

Row 0 is the `[CLS]` token's attention distribution. `[CLS]` has no linguistic meaning of its own, so what it attends to reflects what the layer aggregated as globally important. It's a rough proxy for "sentence-level salience."

---

## Step 6 — Probe coreference with a single token row

```python
it_idx = tokens.index("it")   # find position of "it" in the sequence
it_attn = attn_matrix[it_idx] # grab that entire row

print(f"\nToken 'it' attends most to:")
top_k = np.argsort(it_attn)[::-1][:3]  # top 3 indices, sorted desc
for idx in top_k:
    print(f"  {tokens[idx]:15s}: {it_attn[idx]:.4f}")
```

### What is this probing?

`attn_matrix[it_idx]` is the row for the pronoun "it" — a vector of weights showing how much "it" attends to every other token in this head. If BERT has learned pronoun resolution in this head, you'd expect "it" to attend strongly to "mat" or "cat" (the things it refers to).

In practice: early layers attend mostly to neighbours; middle layers develop semantic awareness; it varies a lot by head. This is exactly what researchers mean when they say certain heads "implement" coreference.

### `np.argsort` explained

```python
np.argsort(it_attn)       # indices that would sort the array ascending
[::-1]                    # reverse → descending
[:3]                      # take top 3
```

---

## Full code (clean version)

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    attn_implementation="eager"
)

# ── 2. Tokenize ───────────────────────────────────────────────────────────────
sentence = "The cat sat on the mat because it was tired"
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)

# ── 3. Forward pass ───────────────────────────────────────────────────────────
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions
print(f"\nLayers : {len(attentions)}")
print(f"Shape  : {attentions[0].shape}")  # (1, 12, seq_len, seq_len)

# ── 4. Single head heatmap ────────────────────────────────────────────────────
layer, head = 0, 0
attn_matrix = attentions[layer][0, head].numpy()

plt.figure(figsize=(10, 8))
plt.imshow(attn_matrix, cmap="Blues")
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.colorbar(label="Attention weight")
plt.title(f"Attention heatmap — Layer {layer}, Head {head}")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()

# ── 5. Average across all heads ───────────────────────────────────────────────
avg_attn = attentions[layer][0].mean(dim=0).numpy()

print("\nAverage attention from [CLS] to each token (layer 0):")
for tok, score in zip(tokens, avg_attn[0]):
    print(f"  {tok:15s} → {score:.4f}")

# ── 6. Coreference probe — what does 'it' attend to? ─────────────────────────
it_idx = tokens.index("it")
it_attn = attn_matrix[it_idx]

print(f"\nToken 'it' attends most to:")
top_k = np.argsort(it_attn)[::-1][:3]
for idx in top_k:
    print(f"  {tokens[idx]:15s}: {it_attn[idx]:.4f}")
```

---

## Things to experiment with

```python
# Loop over all layers and heads to find the most "interesting" head
for l in range(12):
    for h in range(12):
        matrix = attentions[l][0, h].numpy()
        it_row = matrix[tokens.index("it")]
        top_idx = np.argmax(it_row)
        if tokens[top_idx] in ["cat", "mat"]:
            print(f"Layer {l}, Head {h} → 'it' attends to '{tokens[top_idx]}'")

# Compare two sentences
sentence2 = "The dog sat on the mat because it was tired"
# ... run the same pipeline and diff the attention matrices

# Visualise the average across all layers (not just layer 0)
all_layer_avg = torch.stack(attentions).squeeze(1).mean(dim=0).mean(dim=0).numpy()
```

---


## Key concepts summary

| Concept | What it means |
|---------|--------------|
| `attentions[l]` | Attention tensor for layer `l`, shape `(1, 12, seq, seq)` |
| `[0, h, :, :]` | Head `h`'s full attention matrix, shape `(seq, seq)` |
| `matrix[i][j]` | Weight from token `i` to token `j` (row sums to 1.0) |
| `.mean(dim=0)` | Average across heads → single `(seq, seq)` matrix |
| `[CLS]` row | Proxy for sentence-level salience |
| Single token row | Probe what that token "looks at" — useful for coreference |
| `attn_implementation="eager"` | Required to get `output_attentions` back from BERT |
| `torch.no_grad()` | Disables gradient tracking — faster, less memory |
