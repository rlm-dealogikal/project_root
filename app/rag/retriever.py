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

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    embeddings = embedding_fn(documents)

    dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")

    try:
        clusters = dbscan.fit_predict(np.array(embeddings))
    except:
        clusters = [-1] * len(documents)

    for i in range(len(metadatas)):
        metadatas[i]["cluster"] = int(clusters[i])

    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))],
        metadatas=metadatas
    )

    return collection