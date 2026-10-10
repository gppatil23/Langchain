from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
model1 = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')

model2 = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')

prompt1 = PromptTemplate(
    template = 'Generate a short and simple summary of the following text:\n{text}',
    input_variables = ['text'],)

prompt2 = PromptTemplate(
    template = 'Generate a list of 5 questions based on the following text:\n{text}',
    input_variables = ['text'],)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \nNotes:\n{notes}\nQuiz:\n{quiz}',
    input_variables = ['notes','quiz'],)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 |model2 | parser,
})

merged_chain = parallel_chain | prompt3 | model1 | parser

text = '''
Football is a family of team sports in which the object is to get the 
ball over a goal line, into a goal, or between goalposts using merely the body 
(by carrying, throwing, or kicking).[1][2][3]

Unqualified, the word football generally means the form of football 
that is the most popular where the word is used. Sports commonly called football include association football (known as soccer in Australia, Canada, South Africa, the United States, and sometimes in Ireland and New Zealand); Australian rules football; Gaelic football; gridiron football (specifically American football, arena football, or Canadian football); International rules football; rugby league football; and rugby union football.[4] These various forms of 
football share, to varying degrees, common origins and are known as "football codes".

'''
result = merged_chain.invoke({'text':text})
print(result)