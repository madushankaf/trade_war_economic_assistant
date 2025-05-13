# # agent_client.py
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI
# import asyncio
# from dotenv import load_dotenv
# load_dotenv()


# model = ChatOpenAI(model="gpt-4o")


# server_params = StdioServerParameters(
#     command="python",
#     # Make sure to update to the full absolute path to your math_server.py file
#     args=["mcp_rag_agent.py"],
# )

# async def run_agent():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             # Initialize the connection
#             await session.initialize()

#             # Get tools
#             tools = await load_mcp_tools(session)

#             # Create and run the agent
#             agent = create_react_agent(model, tools)
#             agent_response = await agent.ainvoke({"messages": "strategy: Evaluate raising tariffs on rare earth metals on CHina, country USA"})
#             return agent_response

# # Run the async function
# if __name__ == "__main__":
#     result = asyncio.run(run_agent())
#     print(result)

from openai import BaseModel
import uvicorn
from fastapi import FastAPI
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag_agent import evaluate_trade_policy as eval_trade_policy
from dotenv import load_dotenv
from db_utils import get_db_connection, query_previous_strategies
import os

load_dotenv()

router = APIRouter()

class StrategyEvalRequest(BaseModel):
    strategy: str
    country: str
    opposing_country: str = None
    session_id: str

@router.post("/eval-trade-policy")
async def evaluate_trade_policy(request: StrategyEvalRequest):
    """
    Evaluate a trade policy strategy for a given country.
    """
    # Query previous strategies from the database
    previous_strategies = query_previous_strategies(request.session_id, request.country)
    
    # Evaluate the trade policy using the previous strategies
    result = eval_trade_policy(request.strategy, request.country, request.opposing_country, previous_strategies)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to evaluate trade policy")
    return JSONResponse(content=result)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOW_ORIGINS", "http://localhost:3000")],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(router, prefix="/api")

host = "0.0.0.0"
port = 8080
app_name = "main:app"

if __name__ == "__main__":
     uvicorn.run(app_name, host=host, port=port)