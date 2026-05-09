import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

word = "the unicorn sat on the mat because it was hungry"

inputs = tokenizer(word, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)  
    attention = outputs.attentions


print(f"Number of layers: {len(attention)}")
print(f"Shape per layer: {attention[0].shape}")

layer, head = 0, 0
attn_matrix = attention[layer][0, head].numpy()

plt.figure(figsize=(8, 6))
plt.imshow(attn_matrix, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.title(f"Attention weights — layer {layer+1}, head {head+1}")
plt.tight_layout()
plt.show()



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
avg_attn = attention[layer][0].mean(dim=0).numpy()
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

