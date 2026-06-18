import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import streamlit as st
from src.document_loader import load_all_documents
from src.chunker import chunk_documents
from src.embedder import load_embedding_model, embed_chunks
from src.vector_store import get_collection, store_chunks
from src.hybrid_search import HybridSearcher
from src.query_rewriter import rewrite_if_needed
from src.reranker import rerank_chunks
from src.generator import get_llm_client, build_prompt, generate_answer
from src.validator import validate_question
from src.exceptions import ValidationError

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Syllabus Assistant",
    page_icon="🎓",
    layout="centered"
)

# ─── Header ────────────────────────────────────────────────
st.title("🎓 RAG Syllabus Assistant")
st.markdown(
    "Ask any question about your syllabus "
    "and get AI-powered answers with source citations!"
)
st.divider()

# ─── Load pipeline once (cached) ───────────────────────────
@st.cache_resource
def load_pipeline():
    with st.spinner("Loading RAG pipeline..."):
        documents = load_all_documents()
        chunks = chunk_documents(documents)
        model = load_embedding_model()
        embeddings = embed_chunks(model, chunks)
        collection = get_collection()
        store_chunks(collection, chunks, embeddings)

        chunk_texts = [c.page_content for c in chunks]
        chunk_metas = [
            {
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", 0)
            }
            for c in chunks
        ]
        searcher = HybridSearcher(chunk_texts, chunk_metas)
        return model, collection, searcher

model, collection, searcher = load_pipeline()

# ─── Chat history ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat input ────────────────────────────────────────────
if question := st.chat_input("Ask a question about your syllabus..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                # Validate
                validated_q = validate_question(question)

                # Rewrite if needed
                rewritten = rewrite_if_needed(validated_q)

                # Hybrid search
                chunks, metas = searcher.hybrid_search(
                    model, collection, rewritten, top_k=6
                )

                # Rerank
                reranked_chunks, reranked_metas, scores = rerank_chunks(
                    validated_q, chunks, metas
                )

                # Generate
                llm = get_llm_client()
                prompt = build_prompt(
                    validated_q,
                    reranked_chunks[:3],
                    reranked_metas[:3]
                )
                answer = generate_answer(llm, prompt)

                # Show answer
                st.markdown(answer)

                # Show sources
                st.divider()
                st.caption("📚 Sources retrieved:")
                for i, meta in enumerate(reranked_metas[:3]):
                    score = scores[i]
                    st.caption(
                        f"• {meta['source']} — "
                        f"Page {meta['page']} "
                        f"(relevance: {score:.2f})"
                    )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except ValidationError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Info")
    st.metric("Documents", "2")
    st.metric("Chunks", "801")
    st.metric("RAGAS Score", "0.97 / 1.0")
    st.divider()
    st.caption("Built with LangChain, ChromaDB, Groq")
    st.caption("Phase 5 — Production Deployment")