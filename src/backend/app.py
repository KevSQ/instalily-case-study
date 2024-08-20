import os
from dotenv import load_dotenv
from fastapi import FastAPI

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Final

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableWithMessageHistory


from embedding import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from db_manage import get_qdrant_retriever
from HistoryCache import get_session_history, get_user_id, get_session_id, get_store

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

CHUNK_SIZE: Final[int] = EMBEDDING_DIMENSIONS.SMALL.value
COLLECTION_NAME: Final[str] = "partselect_products"
#Most useful for straigtht text completion, not really tuned for much.
# llm = OpenAI(api_key=llm_key)

#TODO: Eventually transition to Qdrant Cloud for deployment
qdrant_client = QdrantClient(url="http://localhost:6333")


if qdrant_client.collection_exists(COLLECTION_NAME) is False:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config= VectorParams(size=CHUNK_SIZE, distance=Distance.COSINE)
    )

qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=EMBEDDING_MODEL
)
retriever = get_qdrant_retriever(qdrant)

template = """You are an AI chatbot assistant for the Ecommerce website PartSelect. Your job is to answer client
questions on inventory selection, part information, what previous reviews have said about the various parts,
and how to install them. You should answer based solely on the context provided:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI chatbot assistant for the Ecommerce website PartSelect. Your job is to answer client
            questions on inventory selection, part information, what previous reviews have said about the various parts,
            and how to install them. You should answer based solely on the context provided: {context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key)

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)
with_message_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history= get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

#TODO: Fix chat history (maybe just implement a database at this rate), see https://github.com/langchain-ai/langchain/discussions/16582
resp = with_message_history.invoke(
    {"question": "What can you tell me about PartSelect item number PS12364199?"},
    config={
        "configurable": {"session_id": "session_id"}
    },
)
print(resp)
print(get_store())
# llm_config = {"model": "gpt-3.5-turbo", "api_key": os.getenv("OPENAI_API_KEY")}
# assistant = AssistantAgent()



# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
