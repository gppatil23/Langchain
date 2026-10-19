from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']

)

prompt2 = PromptTemplate(
    template='Explain the following joke{text}',
    input_variables=['text']
)
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')


parser = StrOutputParser()

chain = RunnableSequence(prompt1,model, parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))