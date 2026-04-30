from transformers import AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "the is the text you have to breakdown for now "
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(tokens)
print(tokens)
print(token_ids)
