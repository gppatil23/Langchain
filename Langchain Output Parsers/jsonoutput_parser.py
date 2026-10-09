from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256,task_type='text-generation')
parser = JsonOutputParser()
template1 = PromptTemplate(
    template = "give me the name,age and city of fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions":parser.get_format_instructions()}

)
chain = template1 | model | parser
final_result = chain.invoke({})  #{} is used to pass input variables to the chain, but in this case, there are no input variables required for the prompt.
print(final_result)
print(type(final_result))