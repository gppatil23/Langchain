#Runnable Passthrough gives same output as you give input

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')



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

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
print(final_chain.invoke({'topic':'Football'}))