from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256,task_type='text-generation')
class Person(BaseModel):
    name: str = Field( description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field( description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "give me the name,age and city of fictional {place} \n {format_instructions}",
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


print(template)  #for understanding the prompt structure and format instructions

chain = template | model | parser
final_result = chain.invoke({"place": "Indian"})
print(final_result)