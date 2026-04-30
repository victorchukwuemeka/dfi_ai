import ollama

prompt = "Explain quantum computing in simple terms for a 10 years old"
model_name = "tinyllama"



#setting our test
termp = [
    {"temp":0.2, "top-p":1.0, "name":"low_temp"},
    {"temp":0.7, "top-p": 1.0, "name":"base_line"},
    {"temp":1.0, "top-p":1.0, "name:":"high_temp"},
    {"temp":0.7, 'top-p':0.8, "name":"low_top_p"},
    {"temp":0.7,  "top-p":0.95, "name":"high_top_p"},
]

for t in termp:
    response = ollama.chat(
        model=model_name,
        messages = [{"role":"user", "content":prompt}],
        options = {"temperature":t["temp"], "top-p":t["top-p"]}
    )

print(f"--- {t['name']} (temp={t['temp']}, top_p={t['top-p']}) ---")
print(response['message']['content'])
print("\n" + "="*80 + "\n")