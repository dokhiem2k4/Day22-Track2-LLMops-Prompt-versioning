"""
Step 1 — LangSmith-instrumented RAG Pipeline
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Load .env BEFORE importing LangChain ─────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-llmops-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── 2. Imports ───────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ── 4. Build FAISS vector store ──────────────────────────────────────────────
def build_vectorstore():
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"  Split into {len(chunks)} chunks")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 5. RAG prompt template ───────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the question using ONLY the context provided below. "
               "If the answer is not in the context, say 'I don't have enough information to answer that.'\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])

# ── 6. Build the RAG chain ───────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 7. Traced query function ─────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# ── 8. Sample questions ──────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the three main types of machine learning?",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "How does regularization prevent overfitting?",
    "What is cross-validation?",
    "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?",
    "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?",
    "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?",
    "What are word embeddings?",
    "What is transfer learning in NLP?",
    "How does BERT handle language understanding?",
    "What is self-attention in transformers?",
    "What is GPT and how is it trained?",
    "What is instruction tuning?",
    "What is RLHF?",
    "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?",
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
    "What is dense retrieval?",
    "Why is chunking strategy important in RAG?",
    "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?",
    "What is FAISS?",
    "How do text embeddings capture semantic meaning?",
    "What is HNSW?",
    "What is hybrid search in vector databases?",
    "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?",
    "What is LangGraph?",
    "What memory types does LangChain support?",
    "What are LangChain retrievers?",
    "What is LangSmith?",
    "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?",
    "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?",
    "What is RAGAS?",
    "How does RAGAS compute faithfulness?",
    "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?",
    "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?",
    "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?",
    "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]

# ── 9. Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    print("\n[1/3] Building vectorstore...")
    vectorstore = build_vectorstore()

    print("[2/3] Building RAG chain...")
    chain, _ = build_rag_chain(vectorstore)

    print(f"[3/3] Running {len(SAMPLE_QUESTIONS)} questions...\n")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:65]}")
        print(f"        A: {answer[:120]}\n")

    print("=" * 60)
    print(f"[OK] {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
