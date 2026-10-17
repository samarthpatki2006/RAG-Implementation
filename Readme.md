RAG Pipeline
This README summarizes the main functions in the RAG pipeline: loading documents, splitting them into chunks, embedding and storing them in Chroma, retrieving relevant chunks, and sending them to the LLM.
Pipeline Overview
Files
  → Load documents
  → Split into chunks
  → Embed document chunks
  → Store chunks and vectors in Chroma
  → Embed the question
  → Retrieve matching chunks
  → Send context and question to the LLM

1. Load Documents
Your load_all_documents(data_dir) helper loads files from the data directory. Depending on the file type, it uses loaders such as PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredExcelLoader, or JSONLoader.
The loader’s .load() method reads a file and returns document objects containing text and metadata.
documents = loader.load()


2. Split Documents into Chunks
Long documents are split into smaller sections so retrieval can find specific information.
If your code uses RecursiveCharacterTextSplitter, the key method is:
chunks = text_splitter.split_documents(documents)
Each chunk retains its text and can retain metadata such as the source filename or page number.


4. Embed Document Chunks
Your EmbeddingPipeline provides the custom method embed_texts(). It converts the text of each chunk into a numerical vector.
texts = [chunk.page_content for chunk in chunks]
document_embeddings = embedding_manager.embed_texts(texts)
These are document embeddings. They are created during indexing, before the chunks are stored.


4. Store Chunks in Chroma
Chroma’s collection.add() method stores each chunk with its ID, text, metadata, and embedding.
vector_store.collection.add(    ids=ids,    documents=texts,    metadatas=metadatas,    embeddings=document_embeddings.tolist())
Each stored record keeps the original chunk text and metadata alongside its vector. Chroma can also create embeddings itself if an embedding function is attached to the collection; the example above supplies the vectors explicitly.


6. Embed the Query and Retrieve Results
RAGRetriever.retrieve() helper first checks whether the collection contains records with Chroma’s collection.count() method.
count = vector_store.collection.count()
It then embeds the question with your EmbeddingPipeline.embed_texts() method and calls Chroma’s collection.query() to find similar stored vectors.
query_embeddings = embedding_manager.embed_texts([query])
results = vector_store.collection.query(query_embeddings=query_embeddings.tolist(),    n_results=min(top_k, count), include=["documents", "metadatas", "distances"])


- query_embeddings supplies the question vector.
- n_results sets the maximum number of matches.
- include specifies which result fields Chroma should return.
- top_k controls how many results the retriever requests.
- score_threshold can filter results according to the score or distance convention used by your code.
Because this call supplies query_embeddings, your code embeds the query itself instead of asking Chroma to embed it.


6. Build Context and Call the LLM
Your rag_advanced() helper collects the retrieved chunk text into a context string and builds a prompt containing the context and question.
It then calls the LLM with:
response = llm.invoke(prompt)
answer = response.content


llm.invoke(): sends the prompt to the model. response.content contains the generated answer.

Main Methods at a Glance

loader.load()	Document loader	Reads a file into document objects
split_documents()	Text splitter, if used in your code	Divides documents into chunks
embed_texts(texts)	Your EmbeddingPipeline	Embeds document chunks or a query
collection.add(...)	Chroma	Stores chunk text, metadata, IDs, and embeddings
collection.count()	Chroma	Checks how many records are stored
collection.query(...)	Chroma	Retrieves records with vectors similar to the query
llm.invoke(prompt)	LangChain chat model	Sends the prompt to the LLM
response.content	LLM response	Reads the generated answer
