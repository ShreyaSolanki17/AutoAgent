"""Quick test: verify Groq API key works and LLM responds.
"""

from backend.llm import call_llm

if __name__ == "__main__":
    print("Sending test request to Groq (llama-3.3-70b-versatile)...\n")

    messages = [
        {"role": "user", "content": "Reply with exactly: 'AutoAgent LLM connection successful.'"}
    ]

    try:
        response = call_llm(messages)
        print(f"[OK] Response: {response}")
    except RuntimeError as e:
        print(f"[ERROR] Config error: {e}")
    except Exception as e:
        print(f"[ERROR] API error: {e}")
