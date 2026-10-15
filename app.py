from src.data_loader import load_all_documents
from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline
from src.search import RAGRetriever
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.simple_rag import rag_simple

load_dotenv()

llm = ChatGroq(
  model="openai/gpt-oss-20b",
  temperature=0.1,
  max_tokens=1024,
) 
if __name__ == "__main__":
    docs = load_all_documents("data")
    store = VectorStore(collection_name="documents_cosine")
    embedding=EmbeddingPipeline()
    retriever=RAGRetriever(vector_store=store,embedding_manager=embedding)

    chunks=embedding.chunk_documents(docs)
    embeds=embedding.embed_chunks(chunks)
    store.add_documents(documents=chunks,embeddings=embeds)

    query:str="What is the capital of Zorblax?"
    results = retriever.retrieve(query=query,top_k=5,score_threshold=0.1)

    # for result in results:
    #     print(f"\nRank: {result['rank']}")
    #     print(f"Similarity: {result['similarity_score']:.4f}")
    #     print(f"Source: {result['metadata'].get('source', 'Unknown')}")
    #     print(f"Content:\n{result['content']}")

    response=rag_simple(query,results,llm,top_k=5)
    print(response)