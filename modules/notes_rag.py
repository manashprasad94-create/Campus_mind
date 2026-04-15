# modules/notes_rag.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Global variables
_embeddings = None
_llm = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

def _load_components():
    """Load components only when first needed"""
    global _embeddings, _llm

    if _embeddings is not None and _llm is not None:
        return _embeddings, _llm

    print("🧠 Loading Notes RAG components...")

    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
)

    print("✅ Notes RAG components ready!")
    return _embeddings, _llm


prompt_template = """
You are a helpful study assistant.
Answer the student's question based ONLY on their notes.
Always respond in the same language the student asks in.
Be concise and clear.
If answer is not in notes say:
"This doesn't appear to be in your notes.
Try asking something else!"

Notes Context:
{context}

Student Question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def create_notes_vectorstore(markdown_text: str):
    embeddings, _ = _load_components()

    docs = [Document(page_content=markdown_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["##", "#", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_notes_answer(question: str, vectorstore) -> str:
    try:
        _, llm = _load_components()

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

        response = chain.invoke({"query": question})
        return response["result"]

    except Exception as e:
        return f"Error: {str(e)}"