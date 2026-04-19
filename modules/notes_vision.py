# modules/notes_vision.py

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_VISION"),
    base_url="https://api.groq.com/openai/v1"
)

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

### 💡 Suggestions
| # | Topic | Suggestion |
|---|-------|------------|

### ✅ Summary
[2 line encouraging summary]

> 🤖 *All recommendations above are AI-generated suggestions.
> Please verify with your textbook or professor.*
---RECOMMENDATIONS_END---
"""


# ✅ SINGLE IMAGE (keep this)
def process_image(image_path: str) -> tuple[str, str]:
    try:
        print("📸 Reading image...")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        mime_type = "image/jpeg"

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content

        return (
            extract_section(raw, "NOTES_START", "NOTES_END"),
            extract_section(raw, "RECOMMENDATIONS_START", "RECOMMENDATIONS_END")
        )

    except Exception as e:
        return f"Error: {str(e)}", ""


# ✅ MULTIPLE IMAGES (THIS IS THE NEW ONE)
def process_multiple_images(image_paths: list[str]) -> tuple[str, str]:
    try:
        print("📸 Reading multiple images...")

        content = [{"type": "text", "text": VISION_PROMPT}]

        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })

        print("🤖 Sending multiple images to Meta...")

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        raw = response.choices[0].message.content

        return (
            extract_section(raw, "NOTES_START", "NOTES_END"),
            extract_section(raw, "RECOMMENDATIONS_START", "RECOMMENDATIONS_END")
        )

    except Exception as e:
        return f"Error: {str(e)}", ""


# ✅ PDF (keep as is)
def process_pdf(pdf_path: str) -> tuple[str, str]:
    try:
        print("📄 Reading PDF...")

        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.pdf",
                                "file_data": pdf_data
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content

        return (
            extract_section(raw, "NOTES_START", "NOTES_END"),
            extract_section(raw, "RECOMMENDATIONS_START", "RECOMMENDATIONS_END")
        )

    except Exception as e:
        return f"Error: {str(e)}", ""


def extract_section(text: str, start_tag: str, end_tag: str) -> str:
    try:
        start = text.index(f"---{start_tag}---") + len(f"---{start_tag}---")
        end = text.index(f"---{end_tag}---")
        return text[start:end].strip()
    except ValueError:
        return text.strip()