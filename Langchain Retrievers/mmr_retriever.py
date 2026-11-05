#Most Importatnt Retriever for Langchain is MMR Retriever which is used to retrieve the most relevant documents from a collection of documents based on a query. It uses a combination of semantic search and keyword matching to find the most relevant documents. The MMR Retriever is implemented in the mmr_retriever.py file.
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv() 

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

docs =[
Document(page_content="LangChain makes it easy to work with LLMs."),
Document(page_content="LangChain is used to build LLM based applications."),
Document(page_content="Chroma is used to store and search document embeddings."),
Document(page_content="Embeddings are vector representations of text."),
Document(page_content="MMR.helps you get diverse results when doing similarity search."),
Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),]

vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)

retriever = vector_store.as_retriever(search_type='mmr',search_kwargs={"k": 3, "lambda_mult": 0.5}) #k=3 means it will return the top 3 most similar documents for a given query
#lambda_mult is a parameter that controls the trade-off between relevance and diversity in the results. A value of 0.5 means that the retriever will return results that are 50% relevant and 50% diverse.
query = "What is Langchain?"
results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"Result{i+1}")
    print(f"content:\n{doc.page_content}...")