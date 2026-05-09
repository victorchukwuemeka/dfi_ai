import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

sentence = "The cat sat on the mat because it was tired"
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# output_attentions=True gives us the attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# attentions: tuple of (num_layers,) each shape [batch, heads, seq, seq]
attentions = outputs.attentions
print(f"Layers: {len(attentions)}")
print(f"Shape per layer: {attentions[0].shape}")  # [1, 12, seq_len, seq_len]

# Visualise Layer 0, Head 0
layer, head = 0, 0
attn_matrix = attentions[layer][0, head].numpy()  # [seq_len, seq_len]

plt.figure(figsize=(10, 8))
plt.imshow(attn_matrix, cmap="Blues")
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.colorbar(label="Attention weight")
plt.title(f"Attention heatmap — Layer {layer}, Head {head}")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()

# Average attention across all heads in layer 0
avg_attn = attentions[layer][0].mean(dim=0).numpy()
print("\nAverage attention from [CLS] to each token (layer 0):")
for tok, score in zip(tokens, avg_attn[0]):
    print(f"  {tok:15s} → {score:.4f}")

# Which tokens does "it" attend to most?
it_idx = tokens.index("it")
it_attn = attn_matrix[it_idx]
print(f"\nToken 'it' attends most to:")
top_k = np.argsort(it_attn)[::-1][:3]
for idx in top_k:
    print(f"  {tokens[idx]:15s}: {it_attn[idx]:.4f}")