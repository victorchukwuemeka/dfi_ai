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
