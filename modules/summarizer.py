# modules/summarizer.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_notes(markdown_text: str) -> str:
    try:
        print("📝 Summarizing notes...")

        # ✅ UPDATED PROMPT (STRICT FORMAT)
        prompt = f"""
You are a precise study assistant.

Convert the notes into EXACTLY 6–8 bullet points.

STRICT RULES:
- Each point must be on a NEW LINE
- Each line must start with "• "
- Each point must be SHORT (max 12–15 words)
- Only ONE idea per point
- Do NOT combine multiple concepts
- Preserve important formulas clearly
- Use simple language
- No extra text before or after ❌

GOOD OUTPUT EXAMPLE:
• A decoder converts binary input into one active output line
• Number of outputs = 2^n where n is number of inputs
• Used in memory chip selection
• 3×8 decoder has 3 inputs and 8 outputs

Notes:
{markdown_text}

Return ONLY the bullet points.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # 🔥 lower = cleaner output
        )

        print("✅ Summary generated!")
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"