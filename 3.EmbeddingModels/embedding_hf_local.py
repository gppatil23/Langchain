from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')

#text = 'Delhi is capital of india'

docs =[
    'This is new india',
    'ai is growing day by day',
    'machine learning is refers to teaching machine actions'
]

vector =embedding.embed_documents(docs)
print(str(vector))

'''
sentence-transformers/all-MiniLM-L6-v2
This is a sentence-transformers model: It maps sentences & paragraphs to a
384 dimensional dense vector space and can be used for tasks like clustering
or semantic search.
'''