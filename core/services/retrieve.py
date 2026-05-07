from core.services.embed import get_embedding
from core.services.ingest import collection

def retrieve(query, k=3):
    query_embedding = get_embedding([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0], results["ids"][0]