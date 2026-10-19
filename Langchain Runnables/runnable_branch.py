from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch

load_dotenv()
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=512,task_type='text-generation')

prompt1 = PromptTemplate(
    template='write a detail report about{topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>200,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,branch_chain)
#OR
# final_chain = report_gen_chain | branch_chain

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))

'''
DEF----->RunnableBranch is a control flow component in LangChain that allows you to conditionally
route input data to different chains or runnables based on custom logic.

It functions like an if/elif/else block for chains - where you define a set of condition functions,
each associated with a runnable (e.g., LLM call, prompt chain, or tool). The first matching
condition is executed. If no condition matches, a default runnable is used (if provided).
'''