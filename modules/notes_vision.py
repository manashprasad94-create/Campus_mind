# modules/notes_vision.py

import os
import base64

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

VISION_PROMPT = """
You are an expert academic assistant and fact-checker.

Your job has TWO parts:

PART 1 - TRANSCRIPTION:
Convert the handwritten notes into clean structured markdown.
Rules:
- Use # for main headings
- Use ## for subheadings  
- Use bullet points for lists
- Use **bold** for important terms
- Preserve all mathematical equations
- Fix obvious spelling mistakes
- Keep all information intact
- Structure it logically

PART 2 - AI RECOMMENDATIONS:
Carefully review the content and organize findings into 
three clear categories:

Format your response EXACTLY like this:

---NOTES_START---
[Clean markdown notes here]
---NOTES_END---

---RECOMMENDATIONS_START---
### ⚠️ Errors Found
| # | You Wrote | Correction | Reason |
|---|-----------|------------|--------|
| 1 | [wrong]   | [correct]  | [why]  |

### 💡 Suggestions
| # | Topic | Suggestion |
|---|-------|------------|
| 1 | [topic] | [suggestion] |

### ✅ Summary
[2 line encouraging summary of notes quality]

> 🤖 *All recommendations above are AI-generated suggestions.
> Please verify with your textbook or professor.*
---RECOMMENDATIONS_END---

IMPORTANT RULES:
- If no errors found skip the Errors table and write: 
  "### ⚠️ Errors Found\n✅ No errors detected!"
- Keep suggestions concise — max 1-2 lines each
- Never alter the student's original meaning
- Always add the AI disclaimer at the end
"""

def process_image(image_path: str) -> tuple[str, str]:
    """
    Takes path to handwritten image
    Returns tuple of (markdown_notes, ai_recommendations)
    """
    try:
        print("📸 Reading image...")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.split(".")[-1].lower()
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        print("🤖 Sending to Gemini Vision...")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=image_data
                            )
                        ),
                        types.Part(text=VISION_PROMPT)
                    ]
                )
            ]
        )

        raw = response.text
        
        # Parse notes and recommendations
        notes = extract_section(raw, "NOTES_START", "NOTES_END")
        recommendations = extract_section(
            raw, 
            "RECOMMENDATIONS_START", 
            "RECOMMENDATIONS_END"
        )

        print("✅ Processing complete!")
        return notes, recommendations

    except Exception as e:
        return f"Error: {str(e)}", ""


def process_pdf(pdf_path: str) -> tuple[str, str]:
    """
    Takes path to PDF
    Returns tuple of (markdown_notes, ai_recommendations)
    """
    try:
        print("📄 Reading PDF...")

        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        print("🤖 Sending to Gemini Vision...")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_data
                            )
                        ),
                        types.Part(text=VISION_PROMPT)
                    ]
                )
            ]
        )

        raw = response.text
        notes = extract_section(raw, "NOTES_START", "NOTES_END")
        recommendations = extract_section(
            raw,
            "RECOMMENDATIONS_START",
            "RECOMMENDATIONS_END"
        )

        print("✅ Processing complete!")
        return notes, recommendations

    except Exception as e:
        return f"Error: {str(e)}", ""


def extract_section(text: str, start_tag: str, end_tag: str) -> str:
    """
    Extracts content between tags
    """
    try:
        start = text.index(f"---{start_tag}---") + len(f"---{start_tag}---")
        end = text.index(f"---{end_tag}---")
        return text[start:end].strip()
    except ValueError:
        # If tags not found return full text
        return text.strip()