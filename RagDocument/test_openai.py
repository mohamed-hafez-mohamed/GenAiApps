from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
# load environment variables from a .env file into your Python application's environment
load_dotenv() 
# Load openai API key into environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embedding_handle = OpenAIEmbeddings(model="text-embedding-3-large")