import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")




prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")


print("=" * 60)
print(f"Prompt: {prompt}\n")


greedy_out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
print("Greedy decoding:")
print(tokenizer.decode(greedy_out[0], skip_special_tokens=True))



beam_out = model.generate(**inputs, max_new_tokens=30, num_beams=5, early_stopping=True)
print("\nBeam search (5 beams):")
print(tokenizer.decode(beam_out[0], skip_special_tokens=True))




for temp in [0.1, 0.7, 0.9]:
    sample_out = model.generate(
        **inputs, max_new_tokens=30,
        do_sample=True, temperature=temp, top_k=0
    )
    print(f"\nTemperature={temp}:")
    print(tokenizer.decode(sample_out[0], skip_special_tokens=True))


topk_out = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_k=50)
print("\nTop-k (k=50):")
print(tokenizer.decode(topk_out[0], skip_special_tokens=True))


topp_out = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.92, top_k=0)
print("\nTop-p (p=0.92):")
print(tokenizer.decode(topp_out[0], skip_special_tokens=True))


print("\n" + "=" * 60)
print("Manual next-token prediction (greedy logic):")
with torch.no_grad():
    logits = model(**inputs).logits          
next_token_logits = logits[0, -1, :]        
probs = torch.softmax(next_token_logits, dim=-1)
top5 = torch.topk(probs, 5)


print(f"Top 5 next tokens after '{prompt}':")
for prob, idx in zip(top5.values, top5.indices):
    token = tokenizer.decode([idx])
    print(f"  '{token:15s}' → {prob.item():.4f}")
