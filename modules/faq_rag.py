import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Globals (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_faq_chain = None
_memory = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are EduAssist, a smart and friendly FAQ assistant for GNIT (Guru Nanak \
Institute of Technology), Kolkata.

Rules you must follow:
1. Answer ONLY using the provided context. Do not add outside knowledge.
2. Always respond in the SAME LANGUAGE the student used to ask.
3. If the answer has a list (e.g. documents, scholarships), present it as a \
clean bullet list.
4. Keep answers concise, accurate, and warm in tone.
5. If the answer is truly not in the context, respond EXACTLY with:
   "I don't have that information in my FAQ yet. Please contact the GNIT \
admissions office directly or call the helpline (Mon–Sat, 10 AM – 5 PM)."
6. Never make up fees, dates, or policy details.

Context:
{context}

Chat History:
{chat_history}

Student Question: {question}

Answer:
"""


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _load_faq_chain() -> ConversationalRetrievalChain:
    """Lazy-load the full RAG chain on first call. Reuses on subsequent calls."""
    global _faq_chain, _memory

    if _faq_chain is not None:
        return _faq_chain

    print("🚀 Initialising EduAssist FAQ RAG pipeline...")

    # --- Guard: check vectorstore exists ---
    index_path = "vectorstore/faq_index"
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "❌ Vectorstore not found at 'vectorstore/faq_index'.\n"
            "   Run  python ingest.py  first to build the index."
        )

    # --- Embeddings (must match what was used during ingest) ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Vectorstore ---
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # --- Retriever ---
    # MMR (Maximal Marginal Relevance) avoids returning near-duplicate chunks.
    # fetch_k=10 → score 10 candidates, return top k=5 diverse ones.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "lambda_mult": 0.7,   # 0 = max diversity, 1 = max relevance
        },
    )

    # --- LLM ---
    # llama-3.3-70b is the best free Groq model for instruction-following.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,          # Low temp = factual, consistent answers
        max_tokens=512,
    )

    # --- Memory: last 4 turns (8 messages) ---
    _memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # --- Chain ---
    _faq_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": _build_prompt()},
        verbose=False,
    )

    print("✅ EduAssist is ready!")
    return _faq_chain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_faq_answer(question: str) -> dict:
    """
    Ask a question and get an answer with source metadata.

    Returns:
        {
            "answer": str,
            "sources": [{"category": str, "q_num": int, "question": str}],
        }
    """
    if not question or not question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    try:
        chain = _load_faq_chain()
        response = chain.invoke({"question": question.strip()})

        answer = response.get("answer", "").strip()

        # Build source list from retrieved chunks (deduplicated by q_num)
        seen = set()
        sources = []
        for doc in response.get("source_documents", []):
            q_num = doc.metadata.get("q_num", -1)
            if q_num not in seen:
                seen.add(q_num)
                sources.append({
                    "category": doc.metadata.get("category", "General"),
                    "q_num": q_num,
                    "question": doc.metadata.get("question", ""),
                })

        return {"answer": answer, "sources": sources}

    except FileNotFoundError as e:
        return {"answer": str(e), "sources": []}
    except Exception as e:
        return {"answer": f"⚠️ Unexpected error: {str(e)}", "sources": []}


def reset_memory():
    """Clear conversation history (e.g. on new session/user)."""
    global _memory
    if _memory:
        _memory.clear()
        print("🔄 Conversation memory cleared.")