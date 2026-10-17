from typing import List, Any, Dict
from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline as EmbeddingManager


class RAGRetriever:
  def __init__(self,vector_store: VectorStore,embedding_manager: EmbeddingManager):
    self.vector_store = vector_store
    self.embedding_manager = embedding_manager

  def retrieve(self,queries: List[str],top_k: int = 5,score_threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
    if isinstance(queries, str):
      raise TypeError("Pass a list of queries, such as [query]")

    if not queries:
      return []

    if any(not isinstance(query, str) or not query.strip()for query in queries):
      raise ValueError("Each query must be a nonempty string")

    if top_k <= 0:
      raise ValueError("top_k must be greater than zero")

    if not -1 <= score_threshold <= 1:
      raise ValueError("score_threshold must be between -1 and 1")

    queries = [query.strip() for query in queries]

    try:
      count = self.vector_store.collection.count()

      if count == 0:
        print("Vector store is empty")
        return [[] for _ in queries]

      query_embeddings = self.embedding_manager.embed_texts(queries)

      results = self.vector_store.collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
      )

      all_results = []

      for query_index, query in enumerate(queries):
        documents = results["documents"][query_index]
        metadatas = results["metadatas"][query_index]
        distances = results["distances"][query_index]
        ids = results["ids"][query_index]

        retrieved_docs = []

        for rank, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances),start=1):
          similarity_score = 1 - distance

          if similarity_score >= score_threshold:
            retrieved_docs.append({
              "id": doc_id,
              "content": document,
              "metadata": metadata or {},
              "similarity_score": similarity_score,
              "distance": distance,
              "rank": rank
            })

        all_results.append(retrieved_docs)

        print(f"Retrieved {len(retrieved_docs)} documents "f"for query: {query}")
      return all_results

    except Exception as e:
        print(f"Failed to retrieve documents: {e}")
        raise