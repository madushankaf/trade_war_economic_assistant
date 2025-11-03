from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import time
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
import glob

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


def load_document(file_path):
    """Load a document based on its file extension."""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return docs
        elif file_path.endswith('.mhtml'):
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
            return docs
        else:
            print(f"Unsupported file type: {file_path}")
            return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def process_docs_folder(docs_folder='./docs'):
    """Process all PDF and MHTML files in the docs folder."""
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(docs_folder, '*.pdf'))
    # Get all MHTML files
    mhtml_files = glob.glob(os.path.join(docs_folder, '*.mhtml'))
    
    all_files = pdf_files + mhtml_files
    print(f"Found {len(all_files)} files to process")
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        # Load the document
        docs = load_document(file_path)
        if not docs:
            print(f"Failed to load {filename}")
            continue
        
        print(f"Loaded {len(docs)} document parts for {filename}")
        
        # Extract text from all document parts
        all_text = [doc.page_content for doc in docs]
        
        # Chunk the text
        all_chunks = []
        for text in all_text:
            if text.strip():  # Only chunk non-empty text
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks for {filename}")
        
        # Skip if chunk count exceeds 200
        if len(all_chunks) > 200:
            print(f"Skipping {filename}: chunk count ({len(all_chunks)}) exceeds 200")
            continue
        
        # Store to Pinecone with metadata including filename
        if all_chunks:
            store_to_pinecone(all_chunks, metadata={"filename": filename, "source": "document"})


if __name__ == "__main__":
    process_docs_folder('./docs')

