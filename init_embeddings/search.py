import os
import time
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()


pine_cone_api_key = os.getenv("PINECONE_API_KEY")
if pine_cone_api_key is None:
    raise ValueError("Pinecone API key not set in environment variables")


open_ai_api_key = os.getenv("OPENAI_API_KEY")
if open_ai_api_key is None:
    raise ValueError("OpenAI API key not set in environment variables")


pinecone = Pinecone(
    api_key=pine_cone_api_key
)

index_name = "openai-embeddings-index-economics"

existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]
if index_name not in existing_indexes:
    pinecone.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        replicas=1,
        shards=1
    )
    while not pinecone.describe_index(index_name):
        print("Waiting for index to be created...")
        time.sleep(5)

index = pinecone.Index(index_name)

embedding = OpenAIEmbeddings(
    openai_api_key=open_ai_api_key,
    model="text-embedding-3-large",
    chunk_size=1
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)

results = vector_store.similarity_search(
    "what are China and US GDPs?",
    k=25
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")