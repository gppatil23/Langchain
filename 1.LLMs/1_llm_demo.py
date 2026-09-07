from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")




# Initialize the model
model_names = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
llm = None
for model_name in model_names:
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=512,
        )
        break
    except Exception as e:
        last_error = e
        print(f" Model '{model_name}' unavailable: {e}")

if llm is None:
    raise RuntimeError(f"Error initializing Google Generative AI with any supported model. Last error: {last_error}")

#promt
try:
    response = llm.invoke([
        HumanMessage(content="What is capital of India?")
    ])
    print(" Model Response:\n", response.content)
    print(response.response_metadata)
except Exception as e:
    print(f"Error during model invocation: {e}")