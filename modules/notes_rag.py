import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

# Global variables
_embeddings = None
_llm = None


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
        temperature=0.2  # lower = better reasoning consistency
    )

    print("✅ Notes RAG components ready!")
    return _embeddings, _llm


# 🔥 UPDATED PROMPT (core fix)
prompt_template = """
You are a smart study assistant.

Your job is to understand notes AND solve problems.

RULES:
1. If the context contains a QUESTION or PROBLEM, you MUST solve it.
2. Show step-by-step reasoning for calculations or derivations.
3. Use formulas from the notes when available.
4. If the notes contain the question but not the solution, solve it using your knowledge.
5. If truly unrelated, say: "This doesn't appear to be in your notes."

Notes Context:
{context}

Student Question:
{question}

Answer (step-by-step if needed):
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# ✅ BETTER CHUNKING (prevents losing questions)
def create_notes_vectorstore(markdown_text: str):
    embeddings, _ = _load_components()

    docs = [Document(page_content=markdown_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,          # increased
        chunk_overlap=100,       # increased
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# 🔍 Detect if it's a problem-solving query
def _is_problem_query(query: str) -> bool:
    keywords = ["find", "solve", "calculate", "derive", "what is", "evaluate"]
    return any(k in query.lower() for k in keywords)


def get_notes_answer(question: str, vectorstore) -> str:
    try:
        _, llm = _load_components()

        # 🔥 Query rewriting (forces solving behavior)
        if _is_problem_query(question):
            enhanced_query = f"""
Solve this problem step-by-step using the notes if relevant.

Question:
{question}
"""
        else:
            enhanced_query = question

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 5}  # increased recall
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )

        response = chain.invoke({"query": enhanced_query})
        return response["result"]

    except Exception as e:
        return f"Error: {str(e)}"