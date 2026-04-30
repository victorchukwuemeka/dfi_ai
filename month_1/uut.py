from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Tokenization is the process of breaking text into tokens."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(tokens))
