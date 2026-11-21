"""
Run once before starting app.py:
python setup_rag.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Resolved relative to this file, not the current working directory, so the
# backend and this script always agree on the location regardless of where they
# are launched from. Defaults to the index committed at the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", REPO_ROOT / "faiss_index")).resolve()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model — local HuggingFace sentence-transformer, no API key required.
    Loaded lazily so importing this module (as app.py does) has no side effects."""
    global _embeddings
    if _embeddings is None:
        print(f"🧠 Loading local embedding model ({EMBEDDING_MODEL_NAME})...")
        print("   (First run will download ~80 MB — subsequent runs use local cache)")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


# --- LCEL INGESTION CHAIN ---

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Step 1: split raw documents into ~800-char overlapping chunks."""
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks created.")
    return chunks


def embed_and_store(chunks: list[Document]) -> FAISS:
    """Step 2: embed each chunk and persist the resulting FAISS index to disk."""
    print("💾 Embedding chunks and building FAISS index...")
    vector_db = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    vector_db.save_local(str(FAISS_INDEX_DIR))
    print(f"   → Vector store saved to: {FAISS_INDEX_DIR}/  (index.faiss + index.pkl)")
    return vector_db


chunking_step = RunnableLambda(chunk_documents)
embedding_and_storage_step = RunnableLambda(embed_and_store)

ingestion_chain = chunking_step | embedding_and_storage_step


def build_vector_db():
    # Imported here so the backend only needs `datasets` when it actually rebuilds the index.
    from datasets import load_dataset

    # 1. Download Dataset from Hugging Face Hub
    print("📥 Downloading Agriculture Dataset from Hugging Face...")
    hf_dataset = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")

    # 2. Convert rows into LangChain Documents
    print("🔄 Converting dataset into LangChain format...")
    documents = []
    # Process the first 2,000 entries for a fast initial setup.
    
    for row in hf_dataset.select(range(2000)):
        text_content = (
            f"Crop Issue/Question: {row['question']}\n"
            f"Solution/Answer: {row['answers']}"
        )
        doc = Document(
            page_content=text_content,
            metadata={"source": "HuggingFace: KisanVaani/agriculture-qa-english-only"},
        )
        documents.append(doc)
    print(f"   → {len(documents)} documents loaded.")
    # you can also remove .select() to load the full dataset (but may take longer time).

    # 3. Run the full ingestion chain: chunk -> embed -> store
    ingestion_chain.invoke(documents)

    print()
    print("✅ Success! RAG database is ready.")
    print("   You can now run:  streamlit run app.py")

if __name__ == "__main__":
    build_vector_db()