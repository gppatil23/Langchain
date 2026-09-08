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
temprature = param, decide how creative or deterministic model will be range(0,2)
Lower values (e.0 - 0.3) - More deterministic and predictable.
Higher values (0.7 - 1.5) - More-tandom, creative, and diverse.  '''
result = model.invoke('what is capital of china?')
print(result.text)  #gets text information only