import chromadb
from core.utils.chunking import chunk_text
from core.utils.file_loader import load_pdf
from core.services.embed import get_embedding

client = chromadb.Client()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")


def ingest_document(document):
    text = load_pdf(document.file.path)
    chunks = chunk_text(text)

    embeddings = get_embedding(chunks)

    ids = [f"{document.id}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return text