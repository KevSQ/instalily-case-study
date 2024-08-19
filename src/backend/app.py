import os
from fastapi import FastAPI
from qdrant_client import QdrantClient
from langchain_openai import OpenAI
from autogen import AssistantAgent, UserProxyAgent
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()
llm = OpenAI()
qdrantDB = QdrantClient(url="http://localhost:6333")


llm_config = {"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
assistant = AssistantAgent()


@app.get("/")
async def root():
    return {"message": "Hello World"}
