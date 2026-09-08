from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.7,
    max_output_tokens=512,
)

response = llm.invoke([HumanMessage(content="Who is captain of indian cricket T20 team?")])
print("🧠 Model Response:\n", response.content)
