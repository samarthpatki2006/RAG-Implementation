from src.data_loader import load_all_documents
from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline
from src.search import RAGRetriever

if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = VectorStore(collection_name="documents_cosine")
    embedding=EmbeddingPipeline()
    retriever=RAGRetriever(vector_store=store,embedding_manager=embedding)

    chunks=embedding.chunk_documents(docs)
    embeds=embedding.embed_chunks(chunks)
    store.add_documents(documents=chunks,embeddings=embeds)

    query:str="False positive handling"
    results = retriever.retrieve(query=query)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"Content:\n{result['content']}")