from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document 
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
#LLMCHainExtractor is used to create a compressor that uses a language model to extract relevant information from documents. It is used in the ContextualCompressionRetriever to compress documents before retrieval.

from dotenv import load_dotenv
load_dotenv() 

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
model = ChatGoogleGenerativeAI(model='models/gemini-3.6-flash',temperature=0.7)


docs = [
Document(page_content=(
"""The Grand Canyon is one of the most visited natural wonders in the world.
Photosynthesis is the process by which green plants convert sunlight into energy.
Milllions of tourists travel to see it every year. The rocks date back millions of years."""
), metadata={"source": "Doc1"}),

Document(page_content=(
"""In medieval Europe, castles were built primarily for defense.
The chlorophyll in plant cells captures sunlight during photosynthesis.
Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
), metadata={"source": "Doc2"}),

Document (page_content=(
"""Basketball was invented by Dr. James Naismith in the late 19th century.
It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
), metadata={"source": "Doc3"}),

Document (page_content=(
"""The history of cinema began in the late 1800s. Silent films were the earliest form.
Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
Modern filmmaking involves complex CGI and sound design."""
), metadata={"source": "Doc4"})] 

vector_store = FAISS.from_documents(docs,embedding_model)
base_retriever =vector_store.as_retriever(search_kwargs={"k": 5})

#creatte a compressor using llm
coompressor =LLMChainExtractor.from_llm(llm=model)

#create a contextual compression retriever
contextual_compression_retriever =ContextualCompressionRetriever(base_retriever=base_retriever,base_compressor=coompressor)

#query
query='What is Photosynthesis?'

results = contextual_compression_retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"---Result--- {i+1}:")
    print(f"Content:\n {doc.page_content}...")
