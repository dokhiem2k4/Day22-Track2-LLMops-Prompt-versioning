"""
Step 2 — Prompt Hub & A/B Routing
"""

import os
import hashlib
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
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 3. Prompt templates ──────────────────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"

# ── 4. Push prompts to Hub ───────────────────────────────────────────────────
def push_prompts_to_hub(client):
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 - concise 2-4 sentence answers")
        print(f"  Pushed V1 -> {url}")
    except Exception as e:
        print(f"  V1 push error: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 - structured expert 3-5 sentence answers")
        print(f"  Pushed V2 -> {url}")
    except Exception as e:
        print(f"  V2 push error: {e}")

# ── 5. Pull prompts from Hub ─────────────────────────────────────────────────
def pull_prompts_from_hub(client):
    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"  Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"  Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"  Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts

# ── 6. A/B routing ───────────────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# ── 7. Build vectorstore ─────────────────────────────────────────────────────
def build_vectorstore(embeddings):
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

# ── 8. Traced A/B query ──────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}

# ── 9. Sample questions (same 50 as step 1) ──────────────────────────────────
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

# ── 10. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

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

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    print("\n[1/4] Pushing prompts to Prompt Hub...")
    push_prompts_to_hub(client)

    print("\n[2/4] Pulling prompts from Prompt Hub...")
    prompts = pull_prompts_from_hub(client)

    print("\n[3/4] Building vectorstore...")
    vectorstore = build_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(f"\n[4/4] Running {len(SAMPLE_QUESTIONS)} questions with A/B routing...\n")
    v1_count, v2_count = 0, 0

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}/{len(SAMPLE_QUESTIONS)}] [prompt-{version_tag}] {question[:55]}")
        print(f"         A: {result['answer'][:110]}\n")

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

    print("=" * 60)
    print(f"  Routing summary: V1={v1_count}  V2={v2_count}  Total={v1_count+v2_count}")
    print(f"  [OK] {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("  Open https://smith.langchain.com to view traces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
