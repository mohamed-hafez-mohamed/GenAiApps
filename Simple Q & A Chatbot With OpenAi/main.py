import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
# load environment variables from a .env file into your Python application's environment
load_dotenv() 
# LangSmith Tracing configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY_QA_CHATBOT_OPENAI"] = os.getenv("LANGCHAIN_API_KEY_QA_CHATBOT_OPENAI")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
## Create a prompt template for the chatbot
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, llm, temperature, max_tokens):
    if not api_key or not api_key.strip():
        return "Error: No API key provided in sidebar"
    os.environ["OPENAI_API_KEY"] = api_key                         # ← make sure it's set
    ## Create the LLM model with the specified parameters
    llm_model = ChatOpenAI(
        model=llm,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=45,                # ← important
        max_retries=2
    )
    ## Create an output parser to format the response as a string
    output_parser = StrOutputParser()
    ## Create the chain by combining the prompt template, LLM model, and output parser
    chain = prompt_template | llm_model | output_parser
    response = chain.invoke({"question": question})
    return response

## streamlit framework for web app
st.title("Simple Q & A Chatbot With OpenAI")
## Sidebar for user input of OpenAI API key, LLM model selection, temperature, and max tokens
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
llm = st.sidebar.selectbox("Select LLM Model:", ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"])
temperature = st.sidebar.slider("Temperature:", min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens:", min_value = 50, max_value = 300, value = 150)
## User input for the question
st.write("Ask a question to the chatbot:")
input_text = st.text_input("You:")
## Generate and display the response when the user inputs a question
if input_text:
    response = generate_response(input_text, api_key, llm, temperature, max_tokens)
    st.write(f"Response from {llm} model:")
    st.write(response)
else:
    st.write("Please enter a question to get a response from the chatbot.")