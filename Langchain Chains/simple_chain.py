from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')

prompt = PromptTemplate(
    template = "generate 5 interesting facts about {topic}",
    input_variables=['topic']
)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"topic": "cricket"})
print(result)
#chain.get_graph().print_ascii() ----------> To visualize the chain structure in ASCII format.