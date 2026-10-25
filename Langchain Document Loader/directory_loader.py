from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
loader = DirectoryLoader(
    path='directory path',
    glob='*.pdf',  #below is table
    loader_cls=PyPDFLoader  #since glob is pdf so PyPDFLoader
)
docs = loader.load()
#docs = loader.lazy_load()  
'''
lazy_load() - for large set of docs.Runs quickly.
fetch->process->delete->fetch another
less memory use
'''
print(len(docs))  #returns number of pages



'''
**************Types of Glob***************
glob pattern            what it loads
'**/*.txt'              All .txt files in folder
'*.pdf'                 All .pdf files in folder
'data/*.csv             All .csv files in folder
'**/*'                  All types of files
'''