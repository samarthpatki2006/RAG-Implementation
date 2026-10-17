from src.data_loader import load_all_documents
from src.vector_store import VectorStore
from src.embeddings import EmbeddingPipeline
from src.search import RAGRetriever
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.simple_rag import rag_simple
from src.enhanced_rag import rag_enhanced
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

    queries:List[str]=["Elaborate the topics under operating systems course"]
    retrieved_results = retriever.retrieve(queries, top_k=3)

    for query, query_results in zip(queries, retrieved_results):
      answer = rag_simple(query, query_results, llm)
      print(f"Question: {query}")
      print(f"Answer: {answer}\n")

    # for query, query_results in zip(queries, retrieved_results):
    #   answer = rag_enhanced(query, query_results, llm)
    #   print(f"Question: {query}")
    #   print(f"Answer: {answer}\n")