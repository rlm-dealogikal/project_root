# /app/rag/retriever.py
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from sklearn.cluster import DBSCAN

from config.settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    MODEL_NAME
)

from app.preprocessing.text_cleaner import clean_text


def build_vector_db(documents, metadatas):
    """
    Builds or updates a ChromaDB collection with embeddings and clusters.
    Reuses the collection if it already exists to avoid recreating it every run.
    
    Args:
        documents (list[str]): List of text documents to add.
        metadatas (list[dict]): List of metadata dicts corresponding to documents.
        
    Returns:
        collection: The ChromaDB collection object.
    """
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    # Try to get existing collection, or create a new one if it doesn't exist
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    
    if not documents:
        # Nothing to add, just return existing collection
        return collection

    # Generate embeddings for the new documents
    embeddings = embedding_fn(documents)

    # Cluster new documents using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
    try:
        clusters = dbscan.fit_predict(np.array(embeddings))
    except:
        clusters = [-1] * len(documents)

    # Add cluster info to metadata
    for i in range(len(metadatas)):
        metadatas[i]["cluster"] = int(clusters[i])

    # Add documents to collection with simple numeric IDs
    # Note: For real-world usage, consider using UUIDs for unique IDs
    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))],
        metadatas=metadatas
    )

    return collection