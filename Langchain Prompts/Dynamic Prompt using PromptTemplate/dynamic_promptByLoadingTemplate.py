#in dynamic_prompt_ui we write template string here we load template.json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt #use to import prompt template.json
from dotenv import load_dotenv
import streamlit as st 
load_dotenv()

st.header('Research Tool')
model = ChatGoogleGenerativeAI(model='models/gemini-3.5-flash-lite',temperature=0.7)

paper_input = st.selectbox("Select Research Paper Name", ["Select ... ", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical","Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})
if st.button('Send'):
    result = model.invoke(prompt)
    st.write(result.text)