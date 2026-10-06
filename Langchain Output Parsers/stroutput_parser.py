from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256,task_type='text-generation')

#1st prompt -> detail report
template1 = PromptTemplate(
    template = "write detail report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template = "write 5 line summary on following text. /n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black holes"})
print(result)