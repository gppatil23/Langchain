#pypdf used mostly for those pdf who has simple textual data only


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('DA Resume.pdf')
docs = loader.load()

print(len(docs))

print(docs[0].page_content)