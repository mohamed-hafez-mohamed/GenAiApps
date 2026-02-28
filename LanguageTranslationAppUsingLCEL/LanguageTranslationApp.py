from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
# load all environment variables
load_dotenv()
# Load OpenAI API key into environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# LangSmith Tracking configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY_LangTrans"] = os.getenv("LANGCHAIN_API_KEY_LangTrans")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

groq_api_key = os.getenv("GROK_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

system_template = "Translate the following into {language}."
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{text_to_translate}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

# Create FastAPI app and add LangServe routes
app = FastAPI(title = "Language Translation App using LCEL and Groq LLM", version = "1.0",
              description = "An app that translates text from one language to another using LangChain's LCEL and Groq LLM.")

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)




