# Generative AI — Elite Developer Track (6 Months)

## Course Overview
A build‑heavy GenAI program for developers who want production‑grade skills. You will design, build, evaluate, and deploy real GenAI systems with clear metrics.

## Outcomes
By the end of the course, you can:
- Build reliable prompt systems and multi‑tool agents
- Ship RAG systems with grounding and evaluation
- Fine‑tune and validate domain models
- Deliver safe, monitored GenAI products in production

## Structure
- Duration: 6 months
- Format: Weekly lecture + lab + assignment
- Milestones: Monthly mini‑projects + final capstone

---

## Month 1 — Foundations and Inference

### Module 1.1: LLM Architecture and Inference
- Goal: Understand how transformers generate text and why outputs vary
- Core topics: tokens, embeddings, attention, layers, decoding
- Lab: compare temperature and top‑p on the same prompt
- Assignment: build a prompt + settings profile for a real task

### Module 1.2: Prompting as Programming
- Goal: Write prompts that behave predictably
- Core topics: role, task, constraints, output schemas, examples
- Lab: convert a vague prompt into a strict schema output
- Assignment: prompt library with 5 reusable templates

### Module 1.3: Reliability and Guardrails
- Goal: Reduce hallucination and enforce structure
- Core topics: validation, retry loops, self‑checks, refusal rules
- Lab: build a prompt validator with failover responses
- Assignment: reliability report on 3 prompts

### Month 1 Mini‑Project
- Build a prompt‑driven assistant with strict outputs and a quality checklist

---

## Month 2 — Tools, Agents, and RAG

### Module 2.1: Tool Use and Function Calling
- Goal: Use tools safely and deterministically
- Core topics: tool schema design, routing, error handling
- Lab: tool‑using assistant with calculator + search
- Assignment: tool chain with logging and retries

### Module 2.2: Agent Patterns
- Goal: Build multi‑step agents that stay on task
- Core topics: ReAct‑style loops, state tracking, guardrails
- Lab: multi‑step agent with plan → act → verify
- Assignment: agent playbook with failure modes

### Module 2.3: Retrieval‑Augmented Generation
- Goal: Build grounded answers over documents
- Core topics: embeddings, vector search, chunking, metadata
- Lab: basic RAG pipeline over a document set
- Assignment: retrieval quality test report

### Module 2.4: Reranking and Grounding
- Goal: Increase relevance and reduce hallucination
- Core topics: rerankers, hybrid search, citations
- Lab: add reranking and sources to RAG
- Assignment: compare baseline vs improved RAG

### Month 2 Mini‑Project
- Production‑grade RAG system with evaluation harness

---

## Month 3 — Adaptation and Fine‑Tuning

### Module 3.1: When to Tune vs RAG
- Goal: Choose the right adaptation strategy
- Core topics: decision framework, data costs, ROI
- Lab: compare prompt vs RAG for a domain task
- Assignment: strategy memo for a real use case

### Module 3.2: Dataset Design
- Goal: Build high‑quality datasets
- Core topics: labeling, guidelines, QA checks
- Lab: create a small labeled dataset
- Assignment: dataset card with quality metrics

### Module 3.3: Parameter‑Efficient Tuning
- Goal: Improve performance with minimal compute
- Core topics: LoRA, adapters, evaluation
- Lab: run a small LoRA experiment
- Assignment: report gains vs baseline

### Month 3 Mini‑Project
- Domain‑tuned model with measurable improvements

---

## Month 4 — Multimodal Systems

### Module 4.1: Vision‑Language Foundations
- Goal: Work with image + text models
- Core topics: image embeddings, OCR, captioning
- Lab: build an OCR + summarization flow
- Assignment: evaluate extraction accuracy

### Module 4.2: Document Intelligence
- Goal: Extract structured data from documents
- Core topics: layout, tables, forms, schema mapping
- Lab: invoice or contract extraction pipeline
- Assignment: schema‑validated extraction report

### Module 4.3: Optional Audio Workflows
- Goal: Add speech to your system
- Core topics: speech‑to‑text, diarization, summarization
- Lab: meeting notes generator
- Assignment: quality comparison across settings

### Month 4 Mini‑Project
- Multimodal workflow with structured outputs

---

## Month 5 — Evaluation, Safety, and Governance

### Module 5.1: Offline Evaluation
- Goal: Measure quality before shipping
- Core topics: rubrics, LLM‑as‑judge, error buckets
- Lab: build an eval suite for a prior project
- Assignment: evaluation report with fixes

### Module 5.2: Online Evaluation
- Goal: Validate in production
- Core topics: A/B tests, metrics, feedback loops
- Lab: design an online eval plan
- Assignment: metric dashboard spec

### Module 5.3: Safety and Security
- Goal: Prevent harmful outputs and data leaks
- Core topics: prompt injection, jailbreaks, red teaming
- Lab: attack your own system and patch it
- Assignment: mitigation plan

### Month 5 Mini‑Project
- Red‑team report and production guardrails checklist

---

## Month 6 — Deployment and Capstone

### Module 6.1: Performance and Cost
- Goal: Keep latency and spend under control
- Core topics: caching, batching, routing, token budgets
- Lab: cost/latency optimization experiment
- Assignment: optimization report

### Module 6.2: Observability and Reliability
- Goal: Operate GenAI systems safely
- Core topics: logging, tracing, alerts, incident response
- Lab: monitoring dashboard for your system
- Assignment: SLO and alert plan

### Module 6.3: UX and Human‑in‑the‑Loop
- Goal: Design trustworthy GenAI experiences
- Core topics: user control, feedback UX, audit trails
- Lab: usability test and iteration
- Assignment: UX findings report

### Capstone
- Build an end‑to‑end GenAI product with metrics and demo

---

## Capstone Suggestions
- Enterprise search assistant with citations
- Analyst copilot with tool actions and audit logs
- Multimodal document intelligence system

## Grading
- Monthly mini‑projects: 50%
- Mid‑course check‑in: 10%
- Capstone: 40%
