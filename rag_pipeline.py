import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq

# ─── Load API key ──────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Load embedding model offline ─────────────────────────
print("Loading embedding model...")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
print("Model ready!")

# ─── Connect to ChromaDB ───────────────────────────────────
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("syllabus")
print(f"Connected! {collection.count()} chunks ready\n")

# ─── Retriever ─────────────────────────────────────────────
def retrieve(question, top_k=3):
    query_vector = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    chunks = results["documents"][0]
    pages = [m["page"] for m in results["metadatas"][0]]
    return chunks, pages

# ─── Prompt builder ────────────────────────────────────────
def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a helpful academic assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the syllabus."

Context:
{context}

Question: {question}

Answer:"""

# ─── Full RAG pipeline ─────────────────────────────────────
def ask(question):
    print(f"\n{'='*55}")
    print(f"Question: {question}")
    chunks, pages = retrieve(question)
    print(f"Retrieved from pages: {pages}")
    prompt = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\nAnswer:")
    print(response.choices[0].message.content)

# ─── Test ──────────────────────────────────────────────────
print("Your RAG system is ready!\n")
ask("What are the types of disasters mentioned in the syllabus?")