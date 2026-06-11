# Instructor Guide: Adaptation and Fine-Tuning

This file contains teacher-facing guidance and slide notes for delivering the module. It is separate from the student-facing content in `03_adaptation_fine_tuning.md`.

## Slides / Teaching Notes

- **Slide 1: "When to Tune vs RAG vs Prompt" (decision framework diagram)**
  Show the three adaptation axes: Prompting (cheapest, fastest), RAG (medium), Fine-Tuning (most expensive). Key message: exhaust cheaper options first. The decision tree helps learners pick the right strategy.

- **Slide 2: "The ROI of Fine-Tuning" (checklist)**
  Walk through the checklist: "Have you tried 5+ prompt variants? Have you tried RAG?" Most teams tune too early. Emphasize that tuning without data is impossible and tuning without eval is dangerous.

- **Slide 3: "What Makes a Good Dataset" (quality pyramid)**
  Quantity at the base, quality at the top. Show the minimum sizes table. Stress: 500 excellent examples beat 5000 noisy ones. The input-output contract concept is critical — define it before collecting data.

- **Slide 4: "Labeling Guidelines" (before and after example)**
  Show a bad guideline ("Write a polite response") vs a good one (with 5+ checkable rules). Demonstrate inter-labeler agreement calculation. Guidelines are living documents — they evolve as you find edge cases.

- **Slide 5: "How LoRA Works" (diagram of low-rank decomposition)**
  Visual: W (frozen, large) + A × B (trainable, tiny). Show parameter count comparison for a 7B model. Key equation: `W' = W + (alpha/r) × A × B`. Explain rank as the "expressiveness dial."

- **Slide 6: "LoRA Configuration" (hyperparameter table)**
  Walk through r, alpha, target_modules, dropout. Show the alpha/r ratio. Best practice: start with q_proj + v_proj at r=8, alpha=16. Monitor loss; increase rank if underfitting, add dropout if overfitting.

- **Slide 7: "The Training Loop" (workflow diagram)**
  Show the full pipeline: load base model → apply LoRA → format data → train → save adapter. Demo the 30-line training script. Point out that only LoRA params are trained — base model stays frozen.

- **Slide 8: "Regression Testing" (evaluation framework)**
  The biggest tuning risk: model gets better at target task but worse at everything else. Show the evaluation framework with multiple test sets. Emphasize: no regression check = incomplete experiment.

- **Slide 9: "QLoRA for Consumer Hardware" (memory comparison)**
  Show memory requirements: full fine-tune (56GB+) vs LoRA (14GB) vs QLoRA (4-8GB). QLoRA makes 7B tuning possible on a single RTX 3090. The tradeoff is ~10% slower training but drastically lower memory.

- **Slide 10: "Mini-Project Requirements" (deliverable checklist)**
  Walk through the rubric: data quality, training correctness, evaluation rigor, documentation. Emphasize that a complete, reproducible pipeline is more valuable than a high score that can't be reproduced.
