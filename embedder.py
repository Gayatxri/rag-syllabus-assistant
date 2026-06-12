from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

loader = PyPDFLoader(r"C:\Users\HP\my_rag_project\data\syllabus.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

sample_chunks = [chunk.page_content for chunk in chunks[:5]]
embeddings = model.encode(sample_chunks)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Each chunk becomes a vector of {embeddings.shape[1]} numbers")
print(f"\nFirst 10 numbers of Chunk 1's vector:")
print([round(float(x), 4) for x in embeddings[0][:10]])

query = "What disasters are covered in Unit 1?"
query_embedding = model.encode([query])
print(f"\nYour question also becomes a vector of {query_embedding.shape[1]} numbers")
print("First 10 numbers:")
print([round(float(x), 4) for x in query_embedding[0][:10]])