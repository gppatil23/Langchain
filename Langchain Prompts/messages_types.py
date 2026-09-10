#To understand types of messages as we store only messages in list in chatbot.py in mini project
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',max_output_tokens=256)

messages = [
    SystemMessage(content='You are a helpful assistent'),
    HumanMessage(content='Tell me about Langchain')
]
result =model.invoke(messages)
messages.append(AIMessage(content=result.text))
print(messages)