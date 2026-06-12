from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/syllabus.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages from syllabus")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print("\n--- First 3 chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk.page_content)} chars):")
    print(chunk.page_content)
    print("-" * 40)