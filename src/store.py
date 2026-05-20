import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
DB_PATH = "database/chroma"
COLLECTION_NAME = "cuad_chunks"

def get_collection():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",
        model_name="text-embedding-3-small"
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# Store all chunks in ChromaDB
def store_chunks(chunks: list[dict]):
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{chunk['contract_id']}__chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "contract_id": chunk["contract_id"],
            "chunk_index": chunk["chunk_index"],
            "char_start": chunk["char_start"],
            "char_end": chunk["char_end"],
            "token_count": chunk["token_count"]
        })

    batch_size = 100
    total = len(ids)

    for i in range(0, total, batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
        print(f"Stored {min(i + batch_size, total)}/{total} chunks")

def search(query: str, n_results: int = 5,
           exclude_contract_id: str = None) -> list[dict]:
    collection = get_collection()

    where = None
    if exclude_contract_id:
        where = {"contract_id": {"$ne": exclude_contract_id}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return chunks