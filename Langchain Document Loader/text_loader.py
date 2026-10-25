from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')
template = PromptTemplate(
    template='Write important 2 lines of following text -\n{text}',
    input_variables=['text']
)
parser = StrOutputParser()
loader = TextLoader('sample.txt',encoding='utf-8')

docs = loader.load()

chain = template | model | parser
print(chain.invoke({'text':docs}))
