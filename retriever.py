from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ─── Load model and ChromaDB ───────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("syllabus")
print(f"Connected! Collection has {collection.count()} chunks\n")

# ─── Retriever function ────────────────────────────────────
def retrieve(question, top_k=3):
    query_vector = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    chunks = results["documents"][0]
    pages  = [m["page"] for m in results["metadatas"][0]]
    return chunks, pages

# ─── Prompt builder ────────────────────────────────────────
def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a helpful academic assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the syllabus."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

# ─── Test with multiple questions ─────────────────────────
questions = [
    "What is UNIT I about in disaster risk reduction?",
    "What are LTPC credits for MX3084?",
    "What are the types of disasters mentioned in the syllabus?",
]
for question in questions:
    print("=" * 55)
    print(f"Question: {question}")
    chunks, pages = retrieve(question)
    print(f"Retrieved from pages: {pages}")
    print("\nTop chunk retrieved:")
    print(chunks[0][:300])
    print("\nPrompt that will be sent to LLM:")
    prompt = build_prompt(question, chunks)
    print(prompt[:400] + "...")
    print()