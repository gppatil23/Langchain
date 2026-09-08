from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="models/gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.7,            
    max_output_tokens=512,
)
'''
temprature = param, decide how output of model everytime when same prompt passed, 
range(0,2)
Lower values (e.0 - 0.3) - Most of time same to same.
Higher values (0.7 - 1.5) - generate different everytime for same promt.  '''
result = model.invoke('what is capital of china?')
print(result.text)  #gets text information only