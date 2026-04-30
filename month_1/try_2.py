from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

prompt = "the future of AI is"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with different temperature
with torch.no_grad():
    low_temp = model.generate(
        **inputs,
        max_length=50,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1,
    )
    high_temp = model.generate(
        **inputs,
        max_length=50,
        temperature=1.0,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1,
    )

print("Low temp:", tokenizer.decode(low_temp[0], skip_special_tokens=True))
print("High temp:", tokenizer.decode(high_temp[0], skip_special_tokens=True))
