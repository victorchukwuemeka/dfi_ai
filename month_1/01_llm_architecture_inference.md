# LLM Architecture and Inference — Full Course Module

## Module Overview
This module gives learners a practical, high‑level understanding of how transformer‑based LLMs are structured and how they generate text at inference time. The emphasis is on intuition, real‑world behavior, and the decisions that affect quality, latency, and cost.

## Target Audience
- Developers and technical professionals
- Comfortable with basic programming concepts (no deep math required)

## Learning Objectives
By the end of this module, learners will be able to:
- Describe the main components of a transformer LLM
- Explain tokenization and why it affects prompting and pricing
- Interpret how attention and layers shape responses
- Compare decoding strategies and select appropriate settings
- Reason about inference tradeoffs: quality vs speed vs cost

---

## Core Definitions (Must‑Know)
- **LLM (Large Language Model)**: A neural network trained on massive text corpora to predict the next token. It generates text by repeatedly choosing the most likely next token given a prompt and its context window.\n- **Embedding**: A numeric vector representation of text (tokens or sequences) that captures semantic meaning. Embeddings let the model compare, retrieve, or reason about similarity in a mathematical space.\n- **Inference**: The process of running a trained model on an input prompt to produce output tokens. Inference includes tokenization, forward pass (logits), and decoding.\n- **Tokenization**: Converting text into tokens (often sub‑word pieces). Tokenization affects cost, context length, and how precisely the model “sees” your input.\n- **Attention**: The mechanism that lets the model weigh which earlier tokens matter most for predicting the next token.

## Time Estimate
- Lecture: 60–90 minutes
- Lab: 60 minutes
- Assignment: 2–3 hours

## Prerequisites
- Basic familiarity with AI or ML concepts
- Comfort reading simple diagrams and pseudo‑code

---

## Lesson Plan (Instructor Guide)

### 1. Warm‑up (10 minutes)
- Ask: “Why do LLMs sometimes ignore instructions?”
- Collect 3–4 hypotheses from learners

### 2. Core Concepts (35–45 minutes)
1. **What an LLM is (in practical terms)**  
   Large Language Models are essentially sophisticated pattern-matching machines trained on vast amounts of text data. They work by predicting the most likely next word or token in a sequence, given the previous context. Unlike traditional databases that store facts, LLMs generate responses based on statistical patterns they've learned during training. For example, if you ask "What is the capital of France?", the model doesn't "know" the answer like a human would; instead, it recognizes that after "capital of France" the most probable next tokens are "is Paris" based on how often that pattern appears in its training data.

2. **Tokenization: words vs subwords, why prompts behave oddly**  
   Tokenization breaks down text into smaller units called tokens, which can be words, parts of words, or even individual characters. Modern LLMs use subword tokenization (like Byte-Pair Encoding) to handle out-of-vocabulary words efficiently. For instance, "unhappiness" might be tokenized as ["un", "happi", "ness"]. This affects prompting because the model sees your input through this tokenized lens. A small change in wording might result in completely different token sequences, leading to unexpected outputs. Additionally, token limits directly impact cost and context length - longer prompts cost more and may get truncated.

3. **Embeddings: text → vectors**  
   Embeddings transform discrete tokens into continuous vector representations in high-dimensional space. Each token gets mapped to a vector (typically 768-4096 dimensions) where similar meanings cluster together. For example, "king" and "queen" would have vectors that are close in this space, while "king" and "apple" would be far apart. These vectors capture semantic relationships, allowing the model to understand analogies like "king is to queen as man is to woman" through vector arithmetic.

4. **Self-attention: how context is weighted**  
   Self-attention is the mechanism that allows the model to focus on relevant parts of the input when generating each output token. For each position in the sequence, it computes attention scores with all other positions, determining how much each token should influence the current prediction. This creates a "weighted context" where important tokens get higher weights. For example, when predicting the next word after "The cat sat on the...", the attention mechanism might heavily weight "cat" and "sat" while downplaying "the".

5. **Transformer blocks: attention + feed-forward layers**  
   A transformer block consists of a multi-head self-attention layer followed by a feed-forward neural network, with residual connections and layer normalization. Multiple such blocks (typically 12-96 layers) are stacked to form the model. Each block refines the representations, allowing the model to capture increasingly complex patterns and relationships in the data.

6. **Context windows: limits and implications**  
   The context window (typically 2048-8192 tokens) limits how much text the model can consider at once. This affects long conversations or documents - information beyond the window gets forgotten. Larger context windows improve coherence but increase computational cost and latency.

7. **Inference pipeline: prompt → tokens → logits → decoded output**  
   The inference process is the step-by-step path from raw user text to the model's final answer. First, the input prompt is split into tokens so the model can process it numerically. Those tokens are converted into embeddings, combined with positional information, and passed through many transformer layers where attention and feed-forward networks build richer contextual representations. At the end of the forward pass, the model produces **logits**, which are raw scores for every possible next token in the vocabulary. A decoding strategy then turns those scores into an actual token choice. That chosen token is appended to the sequence, and the process repeats one token at a time until the model reaches a stop condition such as an end-of-sequence token, a max token limit, or a stop string.

   
   ```text
   +---------+    +-------------+    +------------------+    +------------+    +--------+
   |  Prompt | -> | Tokenization| -> | Embeddings + Pos | -> | Transformer| -> | Logits |
   | (text)  |    | (text→tokens)|    | Encoding        |    | Layers     |    |(scores)|
   +---------+    +-------------+    +------------------+    +------------+    +--------+
                                                           |
                                                           v
                                                +------------------------+
                                                |     Decoding           |
                                                | (greedy / top-k / top-p|
                                                |   / temperature)       |
                                                +------------------------+
                                                           |
                                                           v
                                                +------------------------+
                                                |     Final Output       |
                                                |  (decoded text answer) |
                                                +------------------------+
   ```

   **Step-by-step view of inference**
   - **1. Prompt ingestion**: The system receives raw text such as "Explain photosynthesis simply."
   - **2. Tokenization**: The text is broken into tokens, which may be whole words, subwords, punctuation marks, or special tokens.
   - **3. Embedding lookup**: Each token ID is mapped to a dense vector so the model can operate on numbers instead of text symbols.
   - **4. Positional encoding / position information**: The model adds information about token order so it can distinguish between sequences like "dog bites man" and "man bites dog."
   - **5. Transformer forward pass**: The token representations move through stacked layers of self-attention and feed-forward networks, allowing each token to incorporate context from earlier tokens.
   - **6. Next-token logits**: For the current position, the model outputs a score for every token in its vocabulary. These scores are called logits and are not probabilities yet.
   - **7. Probability conversion**: A softmax function converts logits into probabilities, producing a distribution over possible next tokens.
   - **8. Decoding decision**: The system selects the next token using a rule such as greedy decoding, top-k sampling, top-p sampling, or temperature-scaled sampling.
   - **9. Append and repeat**: The chosen token is added to the sequence, then the model runs again to predict the next one.
   - **10. Detokenization**: Once generation stops, the final token sequence is converted back into human-readable text.

   **Important intuition**
   - The model usually generates output **one token at a time**, not all at once.
   - The logits are just raw scores; decoding is the step that turns those scores into actual text.
   - Small decoding changes can produce very different answers even when the model and prompt stay the same.
   - Longer outputs require more repeated inference steps, which increases latency and cost.

   **Mini example**
   - Prompt: `"The capital of France is"`
   - Likely high-logit candidates for the next token: `" Paris"`, `" Lyon"`, `" Marseille"`
   - After softmax, `" Paris"` may receive the highest probability
   - With greedy decoding, the model picks `" Paris"`
   - The sequence becomes `"The capital of France is Paris"` and generation continues from there


8. **Decoding strategies: greedy, beam, sampling, top-k, top-p**  
   - **Greedy decoding**: Always picks the highest-probability token (deterministic but can be repetitive)  
   - **Beam search**: Keeps track of multiple candidate sequences, choosing the most probable overall path  
   - **Sampling**: Randomly selects from the probability distribution  
   - **Top-k**: Limits sampling to the k most likely tokens  
   - **Top-p (nucleus)**: Samples from the smallest set of tokens whose cumulative probability exceeds p

9. **Temperature and output variability**  
   Temperature scales the logits before applying softmax. Lower temperatures (0.1-0.3) make the distribution more peaked, favoring high-probability tokens for more deterministic outputs. Higher temperatures (0.7-1.0+) flatten the distribution, increasing randomness and creativity but potentially reducing coherence.

### 2a. Real‑World Mental Model (10 minutes)
- **LLMs are "next-token predictors," not fact databases.** They don't store or retrieve facts like a traditional database. Instead, they generate text based on patterns learned from training data. This means they can produce incorrect information if the patterns in their training data were wrong or incomplete.
- **Good prompts reduce ambiguity; decoding settings control creativity vs determinism.** A well-crafted prompt provides clear context and constraints, guiding the model toward desired outputs. Decoding parameters like temperature allow you to balance between creative, diverse responses and consistent, predictable ones.
- **Embeddings are used for search and retrieval; LLMs are used for reasoning and generation.** Embeddings excel at finding similar content or clustering related ideas, while LLMs can perform complex reasoning tasks and generate novel text.
- **Inference is cheap relative to training, but still affects latency and cost in production.** While training a large model can cost millions of dollars and take weeks, running inference is relatively inexpensive. However, factors like model size, context length, and decoding complexity can still impact response times and API costs.

### 3. Practical Intuition (10–15 minutes)
- **Show the same prompt with different temperatures**  
  Example prompt: "Write a short story about a robot learning to paint."  
  - Temperature 0.2: Produces a straightforward, predictable story with conventional plot points  
  - Temperature 0.7: Generates a more creative story with unexpected twists  
  - Temperature 1.2: Creates highly imaginative, potentially surreal content with more variability

- **Compare top‑k vs top‑p for creative vs factual tasks**  
  - Top-k (k=10): Limits choices to the 10 most likely tokens, good for maintaining coherence in factual writing  
  - Top-p (p=0.9): Considers tokens until cumulative probability reaches 90%, allowing more diversity while avoiding very unlikely choices. Better for creative tasks where you want variety but not nonsense.

- **Discuss why lower temperature doesn't guarantee truth**  
  Lower temperature makes outputs more deterministic by favoring high-probability tokens, but this doesn't ensure factual accuracy. If the model's training data contains misinformation or the prompt is misleading, even low-temperature outputs can be incorrect. Temperature affects style and consistency, not inherent truthfulness.

### 4. Wrap‑Up (5 minutes)
- Summarize tradeoffs: accuracy, creativity, latency, and cost
- Preview how this ties into prompting and RAG later

---

> Instructor guide moved to `month_1/01_llm_architecture_inference_instructor.md`.

## Lab: Inference Behavior Playground

### Goal
Experiment with how different inference settings affect output quality, creativity, and consistency using a controlled prompt.

### Materials Needed
- Access to an LLM API (OpenAI, Anthropic, or local model)
- Spreadsheet or notebook for recording results
- Sample prompt: "Explain quantum computing in simple terms for a 10-year-old."

### Steps
1. **Set up your testing environment**  
   Choose your LLM platform and ensure you can modify temperature and top-p parameters.

2. **Run baseline test**  
   Execute the prompt with default settings (usually temperature 0.7-1.0, top-p 1.0) and record the output.

3. **Test temperature variations**  
   - Temperature: 0.2 (very deterministic)  
   - Temperature: 0.7 (balanced)  
   - Temperature: 1.0 (more creative)  
   Record how the explanations differ in simplicity, accuracy, and engagement.

4. **Test top-p variations**  
   - Top-p: 0.8 (focused on high-probability tokens)  
   - Top-p: 0.95 (allows more diversity)  
   Compare how these affect the variety of explanations while maintaining coherence.

5. **Analyze results**  
   - Factual accuracy: Does the explanation remain scientifically correct?  
   - Style consistency: Is the language appropriate for a 10-year-old?  
   - Length control: How does the response length vary?  
   - Creativity: Are there unique analogies or examples?

### Expected Observations
- Lower temperature should produce more consistent, predictable outputs  
- Higher temperature introduces more variation and potentially creative analogies  
- Top-p settings affect diversity without significantly impacting factual accuracy  
- Some combinations may produce "hallucinations" or off-topic content

### Deliverable
Create a one-page report with:  
- Your test prompt  
- Sample outputs for each setting  
- A comparison table rating each output on accuracy, creativity, and appropriateness  
- Recommendations for settings based on different use cases (educational vs. creative writing)

---

## Exercises (Hands‑On Practice)
1. **Define It**  
   Write 2–3 sentences each defining: LLM, embedding, inference, tokenization, attention.  
   *Example for LLM: A Large Language Model is an AI system trained on vast amounts of text data to predict and generate human-like text by learning patterns in language.*

2. **Prompt Compression**  
   Take a 200-word prompt (you can write one or use an existing example) and reduce it to under 80 words while preserving all key requirements. Test both versions with the same model and compare:  
   - Output quality and completeness  
   - Token count and estimated cost  
   - Any loss of important details

3. **Decoding Tradeoffs**  
   Use the prompt: "Write a haiku about artificial intelligence."  
   Run it with:  
   - Temperature 0.2 vs 0.9  
   - Top-p 0.8 vs 0.95  
   Analyze the results and write a short conclusion (3-4 sentences) on when each setting would be appropriate. Consider factors like creativity, traditional form adherence, and uniqueness.

4. **Attention Intuition**  
   Given this prompt: "The weather today is sunny and warm. I decided to go to the beach with my friends. We played volleyball, swam in the ocean, and had a picnic lunch. The waves were gentle and the water was refreshing. As the sun began to set, we headed home tired but happy."  
   Underline the 3 parts most likely to influence the next token prediction. Explain why these specific parts would be weighted heavily by the attention mechanism.

5. **Inference Pipeline Sketch**  
   Draw or describe the step-by-step pipeline from raw text input to final decoded output. Include:  
   - Tokenization process  
   - Model forward pass  
   - Logits computation  
   - Decoding strategy application  
   - Output detokenization

---

## Assignment (Graded)

### Task
Design a complete prompt and inference configuration for a real-world business task. Test your design with at least two different inference settings and analyze the results.

### Example Scenarios
- **Customer Support Response**: Generate helpful, empathetic responses to customer complaints
- **Product Description**: Create compelling product descriptions for an e-commerce site
- **Meeting Summary**: Summarize long meeting transcripts into key action items
- **Code Review Comments**: Generate constructive feedback on code changes

### Requirements
- **Prompt Design**: Include all essential components:  
  - Role definition (e.g., "You are a professional customer service representative")  
  - Clear task description  
  - Specific constraints and guidelines  
  - Output format specification (e.g., "Respond in 2-3 paragraphs with bullet points for action items")  
  - Examples or few-shot learning if appropriate

- **Inference Settings Testing**: Test at least two different configurations:  
  - Setting A: Optimized for accuracy/consistency (e.g., low temperature, focused sampling)  
  - Setting B: Optimized for creativity/variety (e.g., higher temperature, broader sampling)  
  - Run each setting 3-5 times to observe variability

- **Analysis**: Provide justification for your chosen settings based on:  
  - Task requirements (factual vs. creative)  
  - Output quality metrics (accuracy, relevance, engagement)  
  - Practical constraints (latency, cost)  
  - Edge cases and potential failure modes

### Deliverable Format
Submit a document containing:  
1. Task description and chosen scenario  
2. Complete prompt text  
3. Inference settings tested with parameters  
4. Sample outputs from each setting  
5. Comparative analysis and final recommendations  
6. Reflection on what you learned about inference tradeoffs

### Rubric (100 points)
- **Prompt clarity and structure (30 points)**: Well-organized prompt with all necessary components, clear instructions  
- **Quality of output (30 points)**: Outputs meet task requirements, demonstrate appropriate tone and format  
- **Justification of settings (20 points)**: Clear reasoning for parameter choices based on task needs  
- **Reflection and improvement notes (20 points)**: Insights gained, suggestions for optimization, handling of edge cases

---

## Assessment: Quick Quiz (5 Questions)

1. **What is tokenization and why does it matter?**  
   Tokenization is the process of breaking text into smaller units (tokens) that the model can process. It matters because it affects how the model interprets input, influences cost (more tokens = higher cost), and determines context limits.

2. **What is the purpose of attention in a transformer?**  
   Attention allows the model to weigh the importance of different parts of the input when generating each output token, enabling it to focus on relevant context rather than treating all input equally.

3. **How does temperature affect output?**  
   Temperature scales the probability distribution of possible next tokens. Lower temperatures (0.1-0.3) make outputs more deterministic and consistent, while higher temperatures (0.7-1.0+) increase randomness and creativity.

4. **When might you prefer top-p sampling over greedy decoding?**  
   Top-p sampling is preferable when you want a balance between creativity and coherence. Unlike greedy decoding (which always picks the most likely token), top-p allows some diversity by sampling from a nucleus of high-probability tokens, reducing repetitive outputs while avoiding nonsensical choices.

5. **What are the main tradeoffs during inference?**  
   The main tradeoffs are quality vs. speed vs. cost. Higher quality often requires more computation (beam search, larger models), which increases latency and cost. Faster, cheaper inference may sacrifice output quality or creativity.

---

## Common Pitfalls and How to Address Them
- **Mistaking "low temperature" for "truthful output"**  
  Low temperature makes responses more consistent but doesn't guarantee factual accuracy. The model can still generate incorrect information if its training data contained errors. *Solution*: Always validate critical facts and use retrieval-augmented generation (RAG) for factual tasks.

- **Ignoring token limits and truncation**  
  Exceeding context windows leads to silent truncation of input, causing the model to miss important information. *Solution*: Monitor token counts, prioritize key information at the beginning of prompts, and use summarization for long inputs.

- **Overusing creative settings for factual tasks**  
  High temperature or broad sampling can introduce hallucinations in tasks requiring accuracy. *Solution*: Use lower temperatures (0.1-0.3) and focused sampling (top-p 0.8 or top-k 10) for factual content, reserving creative settings for brainstorming or fiction.

- **Assuming the model "knows" facts instead of predicting likely text**  
  LLMs predict statistically likely continuations, not stored knowledge. They can confidently generate wrong information. *Solution*: Treat LLMs as pattern generators, not knowledge bases. Use external verification and fact-checking.

- **Treating embeddings as "small models" rather than vector representations**  
  Embeddings are mathematical vectors, not mini-AI models. They can't perform reasoning or generation on their own. *Solution*: Use embeddings for similarity search and clustering, combine with LLMs for generation tasks.

---

## Resources
- **Papers**: "Attention Is All You Need" (Vaswani et al., 2017) - Original transformer paper
- **Articles**: OpenAI's GPT series documentation, Anthropic's Claude documentation
- **Tools**: Hugging Face Transformers library for experimentation
- **Videos**: 3Blue1Brown's neural network series, Yannic Kilcher's transformer explanations
- **Interactive**: Google Colab notebooks for transformer visualization

---

## Code Examples

### Basic Inference with Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a pre-trained model and tokenizer
model_name = "gpt2"  # or "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define your prompt
prompt = "The future of artificial intelligence is"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with different settings
with torch.no_grad():
    # Low temperature for deterministic output
    outputs_deterministic = model.generate(
        **inputs,
        max_length=50,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1
    )
    
    # Higher temperature for creative output
    outputs_creative = model.generate(
        **inputs,
        max_length=50,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1
    )

# Decode the outputs
text_deterministic = tokenizer.decode(outputs_deterministic[0], skip_special_tokens=True)
text_creative = tokenizer.decode(outputs_creative[0], skip_special_tokens=True)

print("Deterministic output:", text_deterministic)
print("Creative output:", text_creative)
```

### Tokenization Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Tokenization is the process of breaking text into tokens."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(tokens))
```

### Embedding Visualization

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

words = ["king", "queen", "man", "woman", "apple"]
inputs = tokenizer(words, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings.numpy())

print("Cosine similarities:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            print(f"{word1} vs {word2}: {similarities[i][j]:.3f}")
```

---

## Summary
Transformer-based LLMs process tokenized text through attention and layered blocks to generate outputs. Inference settings such as temperature and top-p directly influence variability, accuracy, and cost. Understanding these fundamentals is essential for building reliable GenAI workflows.

## Key Takeaways
- **LLMs are next-token predictors** that generate text based on learned patterns
- **Tokenization** affects how models process input and influences cost
- **Embeddings** capture semantic meaning in vector form
- **Attention** allows models to focus on relevant context
- **Temperature** controls creativity vs consistency in outputs
- **Decoding strategies** offer different tradeoffs between quality, speed, and cost

### Model Architectures Comparison

| Architecture | Primary Use | Key Features | Context Window | Strengths |
|--------------|-------------|--------------|----------------|-----------|
| **GPT (Generative Pre-trained Transformer)** | Text generation | Autoregressive, unidirectional attention | 2048-8192 tokens | Excellent at creative writing, code generation |
| **BERT (Bidirectional Encoder Representations)** | Understanding tasks | Bidirectional attention, masked language modeling | 512 tokens | Strong at classification, question answering |
| **T5 (Text-to-Text Transfer Transformer)** | Various NLP tasks | Encoder-decoder, task-specific prefixes | 512-1024 tokens | Versatile, good at translation and summarization |
| **LLaMA** | General purpose | Efficient attention, large scale | 2048-4096 tokens | Balanced performance, open-source |

### Production Inference Considerations

- **Batching**: Process multiple requests together to improve throughput
- **Quantization**: Reduce model size (FP16, INT8) for faster inference and lower memory usage
- **Caching**: Cache frequent prompts or embeddings to reduce computation
- **Model Parallelism**: Distribute large models across multiple GPUs
- **Latency Optimization**: Use faster decoding methods like speculative decoding

### Common Inference Patterns

1. **Greedy Decoding**: Always pick the most likely token
   ```python
   outputs = model.generate(inputs, do_sample=False, max_length=100)
   ```

2. **Beam Search**: Keep multiple candidates and choose the best overall sequence
   ```python
   outputs = model.generate(inputs, num_beams=5, max_length=100)
   ```

3. **Top-k Sampling**: Sample from the k most likely tokens
   ```python
   outputs = model.generate(inputs, do_sample=True, top_k=50, max_length=100)
   ```

4. **Top-p (Nucleus) Sampling**: Sample from tokens comprising top p probability mass
   ```python
   outputs = model.generate(inputs, do_sample=True, top_p=0.9, max_length=100)
   ```

---

## Extension (Optional)
- **Compare two different models with the same prompt settings**  
  Test GPT-2 vs. GPT-Neo with identical prompts and parameters. Compare output quality, coherence, and computational requirements.

- **Calculate token cost differences for short vs long prompts**  
  Using pricing from OpenAI or Anthropic, calculate costs for prompts of 100, 500, and 2000 tokens. Analyze how prompt length affects total cost per response.

- **Implement a simple RAG system**  
  Build a basic retrieval-augmented generation pipeline using embeddings for document search and LLM for answer generation.

- **Experiment with model quantization**  
  Compare inference speed and output quality between full-precision and quantized models (8-bit, 4-bit).

- **Build a multi-turn conversation system**  
  Implement context management for maintaining conversation history within token limits.

---

## Visual Diagrams

### Transformer Architecture Overview
```
Input Text → Tokenization → Embeddings → Positional Encoding
                    ↓
          Multi-Head Self-Attention
                    ↓
            Feed-Forward Network
                    ↓
          Layer Normalization + Residual
                    ↓
             [Repeat N times]
                    ↓
          Final Linear Layer → Softmax → Output Tokens
```

### Attention Mechanism
```
Query (Q) ──┐
            ├─── Attention Weights ── Context Vector
Key (K) ────┘
Value (V) ──┘

For each position i:
Attention_i = softmax( (Q_i × K^T) / sqrt(d_k) ) × V
```

### Inference Pipeline Flowchart
```
┌─────────────────┐
│   User Prompt   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │  (text → token IDs)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Model Pass    │  (tokens → logits)
│                 │
│ • Embeddings    │
│ • Attention     │
│ • Feed-forward  │
│ • Layer × N     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoding      │  (logits → tokens)
│                 │
│ • Greedy        │
│ • Sampling      │
│ • Temperature   │
│ • Top-k/Top-p   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Detokenization  │  (tokens → text)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Output   │
└─────────────────┘
```

---

## Troubleshooting Common Issues

### Problem: Model generates repetitive or looping text
**Symptoms**: Output repeats phrases, gets stuck in loops  
**Causes**: Too low temperature, insufficient diversity in sampling  
**Solutions**: 
- Increase temperature (0.7-1.0)
- Use top-p sampling instead of greedy decoding
- Add "diversity" instructions to prompt
- Implement repetition penalties

### Problem: Outputs are too random or nonsensical
**Symptoms**: Incoherent responses, off-topic content  
**Causes**: Temperature too high, overly broad sampling  
**Solutions**:
- Decrease temperature (0.3-0.5)
- Use top-k sampling with smaller k (10-20)
- Tighten top-p (0.8-0.9)
- Improve prompt specificity

### Problem: Model ignores prompt instructions
**Symptoms**: Responses don't follow requested format or role  
**Causes**: Insufficient prompt engineering, context dilution  
**Solutions**:
- Use clear role definitions ("You are a...")
- Add explicit format instructions
- Place constraints at beginning of prompt
- Use few-shot examples

### Problem: Slow inference or high latency
**Symptoms**: Long response times, timeouts  
**Causes**: Large models, long contexts, inefficient decoding  
**Solutions**:
- Use smaller/faster models (DistilBERT, TinyLLaMA)
- Implement model quantization (8-bit, 4-bit)
- Reduce context window
- Use greedy decoding for speed
- Enable batching for multiple requests

### Problem: High API costs
**Symptoms**: Expensive usage, budget overruns  
**Causes**: Long prompts, verbose outputs, frequent requests  
**Solutions**:
- Compress prompts while retaining key information
- Set max_tokens limits
- Cache frequent responses
- Use smaller models for simple tasks
- Implement rate limiting

### Problem: Context window limitations
**Symptoms**: Important information gets cut off  
**Causes**: Input exceeds model's context length  
**Solutions**:
- Summarize long inputs first
- Prioritize key information at prompt start
- Use sliding window techniques
- Split long tasks into multiple calls
- Choose models with larger context windows

---

## Real-World Case Studies

### Case Study 1: Customer Support Chatbot
**Challenge**: Build a helpful, consistent customer service bot  
**Solution**: 
- Use low temperature (0.3) for factual accuracy
- Implement strict prompt formatting
- Add validation loops for inappropriate responses
- **Result**: 85% customer satisfaction, reduced support tickets by 40%

### Case Study 2: Code Generation Assistant
**Challenge**: Generate syntactically correct, useful code  
**Solution**:
- Fine-tune on code datasets
- Use medium temperature (0.6) for creativity
- Implement syntax checking in generation loop
- **Result**: 70% of generated code runs without errors

### Case Study 3: Content Summarization
**Challenge**: Create accurate, concise summaries of long documents  
**Solution**:
- Use extractive + abstractive approaches
- Low temperature (0.2) for consistency
- Implement length constraints
- **Result**: 90% factual accuracy, 60% shorter than human summaries

### Case Study 4: Creative Writing Assistant
**Challenge**: Help writers overcome creative blocks  
**Solution**:
- High temperature (0.9) for variety
- Top-p sampling (0.95) for coherence
- Multiple generation attempts for selection
- **Result**: Writers report 3x faster ideation process

---

## Performance Benchmarks

### Inference Speed Comparison (tokens/second)
| Model Size | CPU | GPU (RTX 3080) | A100 |
|------------|-----|----------------|------|
| 125M params | 15 | 120 | 200 |
| 1.3B params | 3 | 25 | 80 |
| 7B params | 0.5 | 8 | 30 |
| 30B params | 0.1 | 2 | 12 |

### Memory Requirements
| Model Size | FP32 | FP16 | INT8 | INT4 |
|------------|------|------|------|------|
| 125M | 500MB | 250MB | 125MB | 62MB |
| 1.3B | 5GB | 2.5GB | 1.25GB | 625MB |
| 7B | 28GB | 14GB | 7GB | 3.5GB |
| 30B | 120GB | 60GB | 30GB | 15GB |

### Cost Analysis (per 1K tokens)
| Provider | Input Cost | Output Cost | Context Window |
|----------|------------|-------------|----------------|
| GPT-4 | $0.03 | $0.06 | 8192 |
| GPT-3.5 | $0.002 | $0.002 | 4096 |
| Claude-2 | $0.008 | $0.024 | 100000 |
| Local (7B) | $0.001 | $0.001 | 4096 |

---

## Future Trends

### Emerging Techniques
- **Speculative Decoding**: Use smaller models to speed up larger model inference
- **Mixture of Experts (MoE)**: Activate only relevant model parts per token
- **Retrieval-Augmented Generation (RAG)**: Combine retrieval with generation for better factual accuracy
- **Fine-tuned Adapters**: Efficient fine-tuning without full model updates

### Hardware Advancements
- **TPUs**: Specialized chips for transformer workloads
- **Neural Processing Units (NPUs)**: AI-optimized hardware
- **Edge AI**: Running LLMs on mobile devices
- **Quantum Computing**: Potential for faster training and inference

### Research Directions
- **Multimodal Models**: Handling text, images, audio together
- **Long Context**: Models with million-token context windows
- **Efficient Training**: Reducing computational requirements
- **Alignment Research**: Making models more helpful and truthful

---

## Key Takeaways

### Core Concepts Mastered
**LLMs are next-token predictors** that generate text by choosing the most likely continuation based on learned patterns  
**Tokenization** breaks text into subword units, affecting prompt effectiveness and cost  
**Embeddings** create vector representations that capture semantic meaning and relationships  
**Attention mechanism** allows models to focus on relevant context when generating each token  
**Transformer blocks** combine attention and feed-forward layers to process and refine representations  

### Inference Mastery
**Temperature** controls output randomness: low (0.1-0.3) for consistency, high (0.7-1.0+) for creativity  
**Top-k and top-p sampling** provide different approaches to controlling output diversity  
**Context windows** limit input length and require careful prompt management  
**Decoding strategies** (greedy, beam, sampling) offer different quality/speed/cost tradeoffs  

### Practical Skills
**Prompt engineering** techniques for reliable, structured outputs  
**Inference optimization** for production deployment (batching, quantization, caching)  
**Cost management** through token efficiency and appropriate model selection  
**Troubleshooting** common issues like repetition, hallucination, and latency  
**Architecture selection** based on use case (GPT for generation, BERT for understanding)
