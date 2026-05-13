Here are the core components you should tell the student to build:

---

## The Inference Observatory — Build List

### 1. **Prompt Input Panel**
- Text area for entering any prompt
- Token counter that updates live as they type
- Estimated cost display before running
- A "Run All" button that fires every strategy simultaneously

### 2. **Side-by-Side Output Columns**
One column per decoding strategy:
- Greedy
- Beam Search (beam size = 3 or 5)
- Top-k Sampling (k = 50)
- Top-p / Nucleus Sampling (p = 0.9)
- Temperature variants (0.2 vs 0.8)

Each column shows the output text, token count, latency in milliseconds, and estimated cost.

### 3. **Token Streaming Visualizer**
- Tokens appear one at a time as they generate
- Each new token briefly highlights so the student can *see* the model building the sentence
- Makes the "one token at a time" concept from the module visceral and real

### 4. **Attention Heatmap**
- Uses a local HuggingFace model (GPT-2 or BERT)
- Click any word in the output and see which input tokens it attended to most
- Color intensity = attention weight
- Directly demonstrates the self-attention concept from the module

### 5. **Consistency Tracker**
- Run the same prompt 5 times per strategy
- Show how much the output varies across runs
- Low temperature = outputs cluster tightly, high temperature = outputs spread apart
- Makes temperature's effect measurable, not just theoretical

### 6. **Inference Tradeoff Dashboard**
A summary table at the bottom showing across all strategies:
- Average latency
- Total tokens used
- Estimated cost
- Output variance score
- So the student can *see* the quality vs speed vs cost tradeoff in one glance

### 7. **Semantic Similarity Score**
- Embed each output using a small embedding model
- Compute cosine similarity between outputs
- Show that greedy and beam search outputs are often semantically close even when the words differ
- Ties embeddings directly to a practical use case

---

## The Tech Stack to Recommend

| Layer | Tool |
|---|---|
| Frontend | React or simple HTML/JS |
| API calls | Anthropic or OpenAI SDK |
| Local model | HuggingFace Transformers (Python) |
| Embeddings | `sentence-transformers` library |
| Attention extraction | HuggingFace with `output_attentions=True` |
| Visualization | D3.js or a simple heatmap library |

---

## What Each Component Teaches

| Component | Module Concept It Proves |
|---|---|
| Token counter | Tokenization and cost |
| Side-by-side columns | Decoding strategies |
| Streaming visualizer | One-token-at-a-time inference |
| Attention heatmap | Self-attention mechanism |
| Consistency tracker | Temperature and output variability |
| Tradeoff dashboard | Quality vs speed vs cost |
| Semantic similarity | Embeddings and vector meaning |

---

