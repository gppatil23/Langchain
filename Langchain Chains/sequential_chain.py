from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')
prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables=['topic'])

prompt2 = PromptTemplate(
    template = 'generate a 5 line summary on the following text. \n {text}',
    input_variables=['text'])

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'Unemployment in India'})
print(result)