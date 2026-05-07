import requests

def generate_answer(context, query):
    prompt = f"""
        You are an AI system.

        Based on context, return JSON:

        {{
        "status": "approved" or "rejected",
        "reason": "short explanation"
        }}

        Context:
        {context}

        Question:
        {query}
        """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]