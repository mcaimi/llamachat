#!/usr/bin/env python

import chroma
from chromadb import Settings, ChromaClient

# ---- RAG Functions ----
def get_chroma_client(chroma_host):
    return chromadb.HttpClient(host=chroma_host.replace("http://", "").replace("https://", ""), settings=Settings())

def retrieve_relevant_docs(query, chroma_host, collection_name, k=3):
    try:
        client = get_chroma_client(chroma_host)
        collection = client.get_collection(name=collection_name)
        results = collection.query(query_texts=[query], n_results=k)
        return results.get('documents', [[]])[0]
    except Exception as e:
        st.warning(f"RAG retrieval failed: {e}")
        return []