import os
from pathlib import Path
from flask import Flask, request, session, jsonify, render_template
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
# To silence the deprecation later, switch to:
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableParallel
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Env & Config

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your .env")

INDEX_DIR = "indexes/attention_faiss"   # must match build_index.py
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 4

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


# Vector store + embeddings

if not Path(INDEX_DIR).exists():
    raise RuntimeError(f"FAISS index not found at {INDEX_DIR}. Run build_index.py first.")

embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},  # cosine-like
)

vectordb = FAISS.load_local(
    folder_path=INDEX_DIR,
    embeddings=embedding,
    allow_dangerous_deserialization=True,
)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K}
).with_config({"run_name": "FAISSRetriever"})


# LLM

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.2)


# Prompt & doc-stuff chain 

messages = [
    SystemMessagePromptTemplate.from_template(
        "You are a precise, citation-friendly assistant.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the answer is not in the context, say you don't know.\n"
        "When citing, use [source i] where i is the chunk number shown in 'Sources'."
    ),
    SystemMessagePromptTemplate.from_template("Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
doc_chain = create_stuff_documents_chain(llm, prompt).with_config({"run_name": "StuffDocsChain"})


# Manual retrieval wiring


# 1) Extract question string and run retrieval
retrieve = RunnableLambda(lambda x: x["question"]) | retriever

# 2) inputs for the doc-stuff chain, INCLUDING chat_history
prep_inputs = RunnableParallel(
    context=retrieve,
    question=RunnableLambda(lambda x: x["question"]),
    chat_history=RunnableLambda(lambda x: x.get("chat_history", [])),  # 👈 pass it through
)

# 3) Final RAG pipeline
rag_chain = RunnableParallel(
    answer=prep_inputs | doc_chain,  # answer string
    context=RunnableLambda(lambda x: x["question"]) | retriever  # List[Document]
).with_config({"run_name": "RAGPipeline"})

# History
_history_store: dict[str, ChatMessageHistory] = {}

def get_session_id() -> str:
    if "uid" not in session:
        session["uid"] = os.urandom(8).hex()
    return session["uid"]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    hist = _history_store.get(session_id)
    if hist is None:
        hist = ChatMessageHistory()
        _history_store[session_id] = hist
    return hist

chain_with_history = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=lambda: get_session_history(get_session_id()),
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def run_config() -> RunnableConfig:
    return RunnableConfig(
        run_name="RAG Ask",
        tags=["flask", "pdf-rag", "groq", "faiss"],
        metadata={"k": TOP_K, "model": LLM_MODEL, "embed_model": EMBED_MODEL},
    )


# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    question = ""

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if question:
            result = chain_with_history.invoke({"question": question}, config=run_config())
            answer = result.get("answer", "")
            docs = result.get("context", []) or result.get("input_documents", []) or []
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                preview = (d.page_content or "")
                if len(preview) > 800:
                    preview = preview[:800] + "..."
                sources.append({
                    "label": f"source {i}",
                    "page": meta.get("page", "N/A"),
                    "source": meta.get("source", "unknown"),
                    "preview": preview,
                })

    # show current chat history
    hist_msgs = get_session_history(get_session_id()).messages
    chat_history = [(m.type, getattr(m, "content", "")) for m in hist_msgs]

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources,
        chat_history=chat_history,
        model_name=LLM_MODEL,
    )

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)


