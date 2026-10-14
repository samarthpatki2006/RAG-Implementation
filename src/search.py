from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline as EmbeddingManager
from typing import List, Any,Dict
class RAGRetriever:
  def __init__(self,vector_store:VectorStore,embedding_manager:EmbeddingManager):
    self.vector_store=vector_store
    self.embedding_manager=embedding_manager

  def retrieve(self,query:str,top_k:int=5,score_threshold:float=0.0)->List[Dict[str,Any]]:
    print(f"Retrieveing documents for the query {query}")
    print(f"Top k: {top_k}, score_threshold: {score_threshold}")
    query_embeddings=self.embedding_manager.embed_texts([query])
    print(query_embeddings)

    try:
      results=self.vector_store.collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=top_k
      )
      retrieved_docs=[]
      if results['documents'] and results['documents'][0]:
        documents=results['documents'][0]
        metadata=results['metadatas'][0]
        distances=results['distances'][0]
        ids=results['ids'][0]

        for i, (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadata,distances)):
          similarity_score=1-distance
          if similarity_score>=score_threshold:
            retrieved_docs.append({
              'id':doc_id,
              'content':document,
              'metadata':metadata,
              'similarity_score':similarity_score,
              'distance':distance,
              'rank':1+i
            })
          
      print(f"Retrieved {len(retrieved_docs)} documents after filtering")
    except Exception as e:
      print(f"Failed to retrieve documents for the given query {query}")
      return []
    return retrieved_docs