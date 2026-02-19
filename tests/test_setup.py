import ollama
from dotenv import load_dotenv

load_dotenv()

from llm.providers import get_llm_provider, resolve_model

# Test Ollama
response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': 'Say hello in exactly 3 words /nothink'}])
print("Ollama:", response['message']['content'][:100])

# Test LLM provider (provider-agnostic)
provider = get_llm_provider()
result = provider.complete(
    messages=[{"role": "user", "content": "Say hello in exactly 3 words"}],
    model=resolve_model("utility"),
    max_tokens=50,
    source="test-setup",
)
print(f"LLM ({resolve_model('utility')}):", result.text)

print("✓ Both LLMs working")
