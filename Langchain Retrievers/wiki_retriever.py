from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever(
    top_k_results=5, lang="en"
)

query = "What is Langchain?"
docs = retriever.invoke(query)

#print required information from the retrieved documents
for i,doc in enumerate(docs):
    print(f"Result {i+1}:")
    print(f"Content:\n {doc.page_content}...")