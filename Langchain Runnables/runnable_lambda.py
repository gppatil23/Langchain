'''
RunnableLambda is a runnable primitive that allows you to apply custom Python functions
within an Al pipeline.

It acts as a middleware between different Al components, enabling preprocessing,
transformation, API calls, filtering, and post-processing in a LangChain workflow.
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a one line joke about {topic}',
    input_variables=['topic']

)

model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256,task_type='text-generation')


parser = StrOutputParser()

def word_count(text):
    return len(text.split())

joke_generate = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})
final_chain = RunnableSequence(joke_generate,parallel_chain)
result = final_chain.invoke({'topic':'AI'})
final_result = """joke - {} \n word count - {}""".format(result['joke'],result['word_count'])
print(final_result)