##This is example of static prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 
load_dotenv()

st.header('Research Tool')
model = ChatGoogleGenerativeAI(model='models/gemini-3.6-flash',temperature=0.7)
user_input = st.text_input('Enter your prompt')
if st.button('Send'):
    result = model.invoke(user_input)
    st.write(result.text)