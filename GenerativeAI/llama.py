from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ ["LANGCHAIN_TRACING_V2" ] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot, so please respond to the user queries properly"),
        ("user", "Question: {question}")
    ]
)

st.title("Langchain Chatbot with Llama3.2")
input_text = st.text_input("What is your question?")
llm = Ollama(model="llama3.2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question" : input_text}))

