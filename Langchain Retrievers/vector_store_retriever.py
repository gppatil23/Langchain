from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
#Source Docs
documents = [
Document(page_content="LangChain helps developers build LLM applications easily."),
Document(page_content="Chroma is a vector database optimized for LLM-based search."),
Document(page_content="Embeddings convert text into high-dimensional vectors."),
Document(page_content="OpenAI provides powerful embedding models."),]

#create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#create vector store
vector_store = Chroma.from_documents(documents=documents, embedding=embedding_model,
                                     collection_name="langchain_docs", persist_directory="./chroma_db")

#create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) #k=2 means it will return the top 2 most similar documents for a given query

query = "What is chroma used for?"
results = retriever.invoke(query)
for i,doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content:\n {doc.page_content}...")
