import os
import re
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def clean_text(text: str) -> str:
    """Remove markdown noise from docling output."""
    text = re.sub(r"#{1,6}\s*", "", text)           # Remove markdown headers
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)  # Remove bold/italic
    text = re.sub(r"-{3,}", "", text)                # Remove horizontal rules
    text = re.sub(r"\n{3,}", "\n\n", text)           # Collapse excess newlines
    return text.strip()


def split_by_question(text: str, source: str) -> list[Document]:
    """
    Split FAQ text into one Document per Q&A pair.
    Matches patterns like: Q.1: , Q.17: , Q.1:- , Q.1 :-
    Each chunk = one full question + its complete answer.
    This is far better than fixed-size chunking for FAQ docs.
    """
    # Regex matches Q.1:, Q.2:-, Q.10: -, etc.
    pattern = r"(Q\.\d+\s*[:\-]+\s*-?\s*)"
    parts = re.split(pattern, text)

    docs = []
    i = 1  # parts[0] is text before first Q (usually empty/header)
    while i < len(parts) - 1:
        question_marker = parts[i].strip()       # e.g. "Q.1:-"
        question_body = parts[i + 1].strip()     # question + answer text

        if not question_body:
            i += 2
            continue

        # Extract question number for metadata
        q_num_match = re.search(r"\d+", question_marker)
        q_num = int(q_num_match.group()) if q_num_match else -1

        # First line is the question, rest is the answer
        lines = question_body.split("\n")
        question_text = lines[0].strip()
        answer_text = "\n".join(lines[1:]).strip()

        full_text = f"{question_marker} {question_text}\n{answer_text}"

        # Detect topic/category from question number
        category = _get_category(q_num)

        docs.append(Document(
            page_content=full_text,
            metadata={
                "source": source,
                "q_num": q_num,
                "question": question_text,
                "category": category,
            }
        ))
        i += 2

    return docs


def _get_category(q_num: int) -> str:
    """Map question numbers to FAQ categories based on PDF structure."""
    if 1 <= q_num <= 16:
        return "Admission"
    elif 17 <= q_num <= 27:
        return "Fees"
    elif 28 <= q_num <= 38:
        return "Hostel"
    elif 39 <= q_num <= 49:
        return "Academics"
    elif 50 <= q_num <= 58:
        return "Exams"
    elif 59 <= q_num <= 67:
        return "Placement"
    elif 68 <= q_num <= 76:
        return "Facilities"
    return "General"


def ingest_faq(pdf_path: str = "data/FAQ.pdf"):
    print("📄 Loading FAQ PDF with Docling...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    raw_markdown = result.document.export_to_markdown()
    print("✅ PDF converted to markdown")

    cleaned_text = clean_text(raw_markdown)

    print("✂️  Splitting into Q&A chunks...")
    chunks = split_by_question(cleaned_text, source=pdf_path)

    if not chunks:
        raise ValueError(
            "No Q&A chunks extracted! Check if FAQ format matches Q.N: pattern."
        )

    print(f"✅ Extracted {len(chunks)} Q&A pairs")

    # Show a preview of extracted chunks
    for c in chunks[:3]:
        print(f"  [{c.metadata['category']}] Q.{c.metadata['q_num']}: "
              f"{c.metadata['question'][:60]}...")

    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}   # Cosine similarity ready
    )
    print("✅ Embedding model loaded")

    print("💾 Creating FAISS vectorstore...")
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/faq_index")
    print(f"✅ Vectorstore saved → vectorstore/faq_index ({len(chunks)} vectors)")

    return vectorstore


if __name__ == "__main__":
    vs = ingest_faq()
    print("\n🔍 Quick retrieval test:")
    results = vs.similarity_search("What is the hostel fee?", k=2)
    for r in results:
        print(f"  [{r.metadata['category']}] {r.page_content[:120]}...")
