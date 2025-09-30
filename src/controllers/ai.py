from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

def ask_ai(analyzed_data: dict, user_message: str | None = None) -> str:
    if user_message:
        prompt = f"""
You are a friendly weather assistant. Here is the confirmed analyzed weather data:
{analyzed_data}

A user has asked: "{user_message}"
Use the analyzed data as the source of truth. Give friendly, human-like guidance.
If the user mentions personal sensitivities (e.g., cold, heat, wind), adjust your advice accordingly for them.
Respond in plain text only. Do not use tables, Markdown, or emojis.
Stay focused on the user's question and the data.
"""
    else:
        prompt = f"""
You are a friendly weather assistant. Focus only on this analyzed data:
Data: {analyzed_data}

Provide concise advice for each activity, mentioning main risks and comfort tips.
Respond in plain text only. Do not use tables, Markdown, or emojis.
Do not deviate from this topic.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=400
    )

    return completion.choices[0].message.content
