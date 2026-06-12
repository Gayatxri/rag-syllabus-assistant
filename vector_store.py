from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

print("Step 1: Loading and chunking syllabus...")
loader = PyPDFLoader(r"C:\Users\HP\my_rag_project\data\syllabus.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

print("\nStep 2: Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready!")

print("\nStep 3: Creating ChromaDB...")
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

try:
    client.delete_collection("syllabus")
except:
    pass

collection = client.get_or_create_collection(
    name="syllabus",
    metadata={"hnsw:space": "cosine"}
)
print("ChromaDB created!")

print("\nStep 4: Embedding and storing all chunks...")
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    ids=[str(uuid.uuid4()) for _ in chunks],
    metadatas=[{"page": chunk.metadata.get("page", 0)} for chunk in chunks]
)
print(f"\nStored {collection.count()} chunks in ChromaDB!")

print("\nStep 5: Testing a search query...")
query = "What disasters are covered in Unit 1?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

print(f"\nTop 3 results for: '{query}'")
print("=" * 50)
for i, doc in enumerate(results["documents"][0]):
    page = results["metadatas"][0][i]["page"]
    print(f"\nResult {i+1} (Page {page}):")
    print(doc[:200])
    print("-" * 40)