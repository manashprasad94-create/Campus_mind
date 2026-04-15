# test.py - Final Integration Test
import sys
sys.path.append(".")

from api import (
    get_faq_answer,
    process_image,
    create_notes_vectorstore,
    get_notes_answer,
    summarize_notes,
    markdown_to_pdf
)

print("=" * 50)
print("🔥 Final Integration Test")
print("=" * 50)

# Test 1 - FAQ
print("\n1️⃣ FAQ Test")
answer = get_faq_answer("Is hostel mandatory?")
print(f"✅ {answer}")

# Test 2 - Vision
print("\n2️⃣ Vision Test")
notes, recommendations = process_image("data/testimg.jpg")
print(f"✅ Notes length: {len(notes)} chars")

# Test 3 - Notes RAG
print("\n3️⃣ Notes RAG Test")
vs = create_notes_vectorstore(notes)
answer = get_notes_answer("What is the main topic?", vs)
print(f"✅ {answer}")

# Test 4 - Summarizer
print("\n4️⃣ Summarizer Test")
summary = summarize_notes(notes)
print(f"✅ Summary length: {len(summary)} chars")

# Test 5 - PDF
print("\n5️⃣ PDF Test")
pdf = markdown_to_pdf(notes, recommendations)
print(f"✅ PDF size: {len(pdf)} bytes")

print("\n" + "=" * 50)
print("🎉 All systems ready for frontend integration!")
print("=" * 50)