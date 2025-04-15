from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import time
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from uuid import uuid4
from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def chunk_text(text):
    return splitter.split_text(text)


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

existing_indexes = [inex_info["name"] for inex_info in pinecone.list_indexes()]
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

def store_to_pinecone(docs, metadata={}):
    print(f"Storing {len(docs)} documents to Pinecone")
    documents = [Document(page_content=doc, metadata=metadata) for doc in docs]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(
        documents=documents,
        ids=uuids
    )


def load_country_topics(country: str, topic_keywords=None):
    if topic_keywords is None:
        topic_keywords = ["economy", "trade", "GDP", "inflation", "trade balance", "exports", "imports"]

    all_docs = []

    for keyword in topic_keywords:
        query = f"{keyword} of {country}"
        try:
            loader = WikipediaLoader(query=query, lang="en", load_max_docs=1)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Skipping {query}: {e}")

    return all_docs


list_of_countries_or_regions = [
    "United States",
    "China",
    "Germany",
    "Japan",
    "India",
    "Brazil",
    "United Kingdom",
    "France",
    "Italy",
    "Canada",
    "Australia",
    "South Korea",
    "Russia",
    "Mexico",
    "European Union",
    "Saudi Arabia",
    "Turkey",
    "Indonesia",
    "Sri Lanka"
]

for country in list_of_countries_or_regions:
    print(f"Loading topics for {country}")
    docs = load_country_topics(country)
    print(f"Loaded {len(docs)} documents for {country}")
    all_text = [doc.page_content for doc in docs]
    for text in all_text:
        print(text)

    all_chunks = []
    for text in all_text:
        print(f"Chunking text for {country}")
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    print(f"Chunked {len(all_chunks)} documents for {country}")
    store_to_pinecone(all_chunks, metadata={"country": country})

    # split_docs = chunk_text(docs)
    # for doc in split_docs:
    #     doc.metadata["country"] = country

    #store_to_pinecone(split_docs, metadata={"country": country})



