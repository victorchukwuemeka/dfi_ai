from transformers import AutoTokenizer
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "this is the text we want to use "
print(" Text:", text)

tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

token_ids = tokenizer.encode(tokens)
print("Token IDs:", token_ids)
