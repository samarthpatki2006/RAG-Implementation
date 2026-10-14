Multi-Format Document Retrieval
A Python project that loads documents, splits their text into chunks, generates sentence embeddings, and stores them in a persistent Chroma vector database. A text query retrieves the most relevant chunks using semantic similarity.
This project implements the retrieval stage of Retrieval-Augmented Generation (RAG). It does not yet call an LLM or generate answers from the retrieved context.
Features
- Recursively discovers supported files inside a data directory.
- Loads PDF, TXT, CSV, Excel, Word, and JSON files.
- Splits documents into overlapping text chunks while retaining metadata.
- Generates embeddings using all-MiniLM-L6-v2 through Sentence Transformers.
- Stores embeddings, text, and metadata locally with Chroma.
- Retrieves the top matching chunks with an optional similarity threshold.
- Returns chunk IDs, content, metadata, distances, similarity scores, and ranks.
How It Works
1. Load: Convert files into LangChain documents containing page_content and metadata.
2. Chunk: Split text using RecursiveCharacterTextSplitter.
3. Embed: Convert each chunk into a numerical vector.
4. Store: Save vectors alongside their text and metadata in Chroma.
5. Search: Embed the query with the same model and retrieve nearby vectors.
6. Filter: Keep matches whose similarity meets the configured threshold.
Project Structure
Organize the supplied code into the following files. main.py is the suggested name for the entry-point script.
main.py
src/
    __init__.py
    data_loader.py
    embeddings.py
    vector_store.py
    search.py
data/
    sample.pdf
    notes.txt
    records.csv
    workbook.xlsx
    report.docx
    messages.json
    vector_store/        # Created by Chroma
README.md
Module	Responsibility
data_loader.py	Defines load_all_documents(data_dir) and file-specific loaders.
embeddings.py	Defines EmbeddingPipeline for chunking and embedding.
vector_store.py	Defines VectorStore for persistence and insertion.
search.py	Defines RAGRetriever for query embedding and retrieval.
main.py	Runs loading, indexing, and an example search.
