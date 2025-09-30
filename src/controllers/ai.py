from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

def ask_ai(analyzed_data: dict, user_message: str | None = None) -> str:
    if not user_message:
        prompt = f"""
You are a friendly weather assistant. Focus only on this analyzed data.
Data: {analyzed_data}

Provide concise advice for each activity, mentioning main risks and comfort tips.
**Respond only in plain text. Do not include tables, Markdown, or emojis.**
Do not deviate from this topic.
"""
    else:
        prompt = f"""
You are a friendly weather assistant. The following is confirmed analyzed weather data:
{analyzed_data}

A user has asked: "{user_message}"
Use the analyzed data as source of truth. Respond in a **friendly, human-like manner** with helpful advice or guidance.
**Respond only in plain text. Do not include tables, Markdown, or emojis.**
Do not recalculate anything; focus on being conversational and practical.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=400
    )

    return completion.choices[0].message.content
