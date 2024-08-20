from langchain_community.document_loaders import FireCrawlLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from embedding import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Final
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_SITE: Final[str] = "https://www.partselect.com/Refrigerator-Parts.htm"
COLLECTION_NAME: Final[str] = "partselect_products"
CHUNK_SIZE: EMBEDDING_DIMENSIONS = EMBEDDING_DIMENSIONS.SMALL.value
qdrant_client: Final[QdrantClient] = QdrantClient(url="http://localhost:6333")

if qdrant_client.collection_exists(COLLECTION_NAME) is False:
    print("partselect_products collections doesn't exist, creating now")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config= VectorParams(size=CHUNK_SIZE, distance=Distance.COSINE)
    )

qdrant: QdrantVectorStore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=EMBEDDING_MODEL
)
def get_qdrant_retriever(vectorStore: QdrantVectorStore):
    if qdrant_client.get_collection(COLLECTION_NAME).points_count == 0:  #If collection is empty, seed
        print("Seeding database with partselect data")
        loader: Final[FireCrawlLoader] = FireCrawlLoader(api_key=os.getenv("FIRECRAWL_API_KEY"), url=CLIENT_SITE, mode="scrape")
        scraped_docs: list = loader.load()
        text_split: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
        all_split_chunks: list = text_split.split_documents(scraped_docs)
        #Seed vector embeddings into database
        vectors = qdrant.from_documents(
            collection_name=COLLECTION_NAME,
            documents=all_split_chunks,
            embedding=EMBEDDING_MODEL
        )
    else:
        vectors = vectorStore

    return vectors.as_retriever()

#TODO: Next, introduce secondary cache db using MySql and use RunnableWithMessageHistory


