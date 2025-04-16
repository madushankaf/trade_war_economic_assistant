from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import time
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import json
import re

from mcp.server.fastmcp import FastMCP


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

retriever = vector_store.as_retriever( search_type="similarity", search_kwargs={"k": 5} )

llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Optional: useful for audit/logs
)

def evaluate_trade_policy(strategy, country):
    prompt = f"""
                You are an economic policy assistant. Based on retrieved reports and expert data,
                evaluate the following trade policy and output ONLY valid JSON.

                Policy: {strategy}
                Country: {country}

                Return exactly this structure:

                {{
                "strategy": "{strategy}",
                "country": "{country}",
                "GDP_change": <numeric, e.g. -0.5>,
                "Political_boost": <numeric scale -5 to +5>,
                "Trade_balance_shift": <numeric, e.g. +0.5>,
                "confidence": "<Low | Medium | High>",
                "source_refs": []
                }}

                Rules:
                - Use only the information from the retrieved documents
                - Do not include any other text or explanation
                - Output only the JSON (no explanation or prose)
                - Returning numeric values is mandatory for GDP_change, Political_boost, Trade_balance_shift
                - Use a scale of -5 to +5 for Political_boost
                - Use a scale of -100% to +100% for Trade_balance_shift
                - Use a scale of -5 to +5 for GDP_change
                - Use historical logic and sources
                - Rationalize your choices in the "rational" field
                - Use the "confidence" field to indicate your certainty about the prediction
                - Use the "source_refs" field to include references to the sources used
                - The "source_refs" field should be a list of strings, each string being a source reference
                - The JSON should be valid and parsable
                - Do not include any other text or explanation
                - Do not include any other keys or values

                Example output:
                {{
                "strategy": "Increase tariffs on imports from India",
                "country": "USA",
                "GDP_change": -0.5,
                "Political_boost": 2,
                "Trade_balance_shift": 0.5,
                "confidence": "Medium",
                "rational": "This policy is expected to reduce imports from India, leading to a slight increase in GDP due to reduced trade deficit. However, it may also lead to retaliation from India.",
                "source_refs": ["source1", "source2"]
                }}
                    """.strip()

    # Invoke the chain
    response = qa_chain.invoke({"query": prompt})
    raw_result = response["result"]

    # Try to extract clean JSON block using regex
    match = re.search(r"\{[\s\S]*\}", raw_result)
    if not match:
        print("⚠️ Failed to find JSON block. Raw result:\n", raw_result)
        return None

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print("⚠️ Failed to parse JSON. Raw string:\n", json_str)
        return None


    return parsed

# Example usage
# result = evaluate_trade_policy("impose 40% tariffs on Chinese imports", "USA")
# print(result)

mcp = FastMCP("macroeconomic_impact")

@mcp.tool()
async def evaluate_trade_policy_tool(strategy: str, country: str) -> dict:
    """
    Evaluate the trade policy and return a structured JSON response.
    """
    result = evaluate_trade_policy(strategy, country)
    if result is None:
        return {"error": "Failed to evaluate trade policy"}
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")