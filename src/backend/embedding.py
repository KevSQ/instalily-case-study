import os
from langchain_openai import OpenAIEmbeddings
from typing import Final
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL: Final[OpenAIEmbeddings] = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small")

class EMBEDDING_DIMENSIONS(Enum):
    SMALL = 1536
    LARGE = 3072