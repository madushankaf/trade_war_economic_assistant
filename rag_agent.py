from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import time
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
import re

#from mcp.server.fastmcp import FastMCP


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

def evaluate_trade_policy(strategy, country, other_country, previous_strategies):
    prompt = f"""
                You are an economic policy assistant. Based on retrieved reports and expert data,
                evaluate the following trade policy and output ONLY valid JSON.

                Policy: {strategy}
                Country: {country}
                Opposing country: {other_country}

                Previous Strategies:
                {previous_strategies if previous_strategies else "No previous strategies available."}

                Allowed strategies (with type):
                - OPEN DIALOGUE (cooperative)
                - RAISE TARIFFS (defective)
                - WAIT AND SEE (cooperative)
                - SANCTION (defective)
                - SUBSIDIZE EXPORT (cooperative)
                - IMPOSE QUOTA (defective)

                If the provided policy is not EXACTLY one of the allowed strategy names above,
                return the following JSON instead:
                {{"error": "Unsupported strategy", "allowed": [
                    "OPEN DIALOGUE", "RAISE TARIFFS", "WAIT AND SEE",
                    "SANCTION", "SUBSIDIZE EXPORT", "IMPOSE QUOTA"
                ]}}

                Otherwise, return exactly this structure:

                {{
                "strategy": "{strategy}",
                "strategy_type": "<cooperative | defective>",
                "country": "{country}",
                "GDP_change": <numeric, e.g. -0.5>,
                "Political_boost": <numeric scale -5 to +5>,
                "Trade_balance_shift": <numeric, e.g. +0.5>,
                "confidence": "<Low | Medium | High>",
                "rationale": "<At least 120 words (3–8 sentences). Must explicitly cover: (1) primary transmission channels: prices, volumes, FX, productivity; (2) most affected sectors and stakeholders; (3) short-term (0–12m) vs medium-term (1–3y) impacts on GDP and trade balance with directionality; (4) retaliation/spillover risks and diplomatic dynamics; (5) 1–2 historical precedents from retrieved sources; (6) key assumptions and the largest uncertainties. Write as a single coherent paragraph, no bullet points>",
                "source_refs": []
                }}

                Rules:
                - Use only the information from the retrieved documents
                - Output only the JSON (no explanation or prose)
                - Returning numeric values is mandatory for GDP_change, Political_boost, Trade_balance_shift
                - Use a scale of -5 to +5 for Political_boost and GDP_change
                - Use a scale of -100% to +100% for Trade_balance_shift (positive improves trade balance)
                - Determine "strategy_type" exactly from the allowed list above
                - The "rationale" must be specific, evidence-grounded, and ≥120 words, covering channels, sectors, time horizon, retaliation, precedents, and assumptions as described above
                - The "source_refs" field should be a list of strings with brief citations
                - The JSON must be valid and parsable with no trailing text
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

# mcp = FastMCP("macroeconomic_impact")

# @mcp.tool()
# async def evaluate_trade_policy_tool(strategy: str, country: str) -> dict:
#     """
#     Evaluate the trade policy and return a structured JSON response.
#     """
#     result = evaluate_trade_policy(strategy, country)
#     if result is None:
#         return {"error": "Failed to evaluate trade policy"}
#     return result

if __name__ == "__main__":
    mcp.run(transport="stdio")