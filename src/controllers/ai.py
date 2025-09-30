from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

def ask_ai(analyzed_data: dict, user_message: str | None = None) -> str:
    prompt = f"""
You are a friendly weather assistant. Focus only on this analyzed data.
Data: {analyzed_data}

Provide concise advice for each activity, mentioning main risks and comfort tips. Do not deviate from this topic.
"""
    if user_message:
        prompt += f"\nUser follow-up: {user_message}"

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=400
    )

    return completion.choices[0].message.content
