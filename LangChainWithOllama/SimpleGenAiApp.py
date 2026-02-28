import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
# load environment variables from a .env file into your Python application's environment
load_dotenv() 
# LangSmith Tracing configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt template for translation
prompt = ChatPromptTemplate.from_messages(
    [
        # System message: tell the llm model its role
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        # Human message: the user input
        ("user", "Question : {question}")
    ]
)

## streamlit framework for web app
st.title("LangChain demo with Ollama llama2 model")
input_text = st.text_input("Enter your question here:")

## Create llama2
llm = Ollama(model = "llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

## Process the input and get the response
if input_text:
    response = chain.invoke({"question": input_text})
    st.write("Response from Ollama Llama2 model:")
    st.write(response)