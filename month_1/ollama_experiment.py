import ollama

prompt = "Explain quantum computing in simple terms for a 10-year-old."
#model_name = "llama3"  # or "llama3.2", "phi3", "mistral"
model_name = "tinyllama"

settings_to_test = [
    {"temp": 0.2, "top_p": 1.0, "name": "Low Temperature"},
    {"temp": 0.7, "top_p": 1.0, "name": "Baseline"},
    {"temp": 1.0, "top_p": 1.0, "name": "High Temperature"},
    {"temp": 0.7, "top_p": 0.8, "name": "Low Top-p"},
    {"temp": 0.7, "top_p": 0.95, "name": "High Top-p"},
]

print(f"Running experiment with model: {model_name}\n")

for setting in settings_to_test:
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": setting["temp"],
            "top_p": setting["top_p"],
        }
    )
    
    print(f"--- {setting['name']} (temp={setting['temp']}, top_p={setting['top_p']}) ---")
    print(response['message']['content'])
    print("\n" + "="*80 + "\n")