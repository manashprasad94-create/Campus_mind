# modules/summarizer.py

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_notes(markdown_text: str) -> str:
    try:
        print("📝 Summarizing notes...")

        prompt = f"""
You are a helpful study assistant.
Summarize the following notes into maximum 8 clear bullet points.
Rules:
- Only include the most important concepts
- Keep each bullet point concise (1 line)
- Use simple easy to understand language
- Preserve any important formulas or definitions
- Start each point with a relevant emoji

Notes:
{markdown_text}

Return ONLY the bullet points, nothing else.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        print("✅ Summary generated!")
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"