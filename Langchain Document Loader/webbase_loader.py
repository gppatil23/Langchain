from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')
template = PromptTemplate(
    template='Answer the following question\n {question}from the following text -\n{text}',
    input_variables=['question','text']
)
parser = StrOutputParser()

url = 'https://www.flipkart.com/samsung-galaxy-s25-ultra-5g-titanium-silverblue-256-gb/p/itm413a5c3f30151?pid=MOBH8K8U5FWQWD6G&lid=LSTMOBH8K8U5FWQWD6GZUGQGG&hl_lid=&marketplace=FLIPKART&fm=eyJ3dHAiOiJwbXVfdjMiLCJwcnB0IjoiaHAiLCJtaWQiOiJjb250aW51dW0vaHAifQ%3D%3D&pageUID=1792941772452'
loader = WebBaseLoader(url)
docs = loader.load()

chain = template | model | parser

print(chain.invoke({'question':'What is weight of this product ?','text':docs[0].page_content}))