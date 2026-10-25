from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path='csv file path')
docs = loader.lazy_load()
print(docs[0])