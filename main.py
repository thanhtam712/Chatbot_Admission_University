import time
import uvicorn
import chainlit as cl
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from chainlit.utils import mount_chainlit

from pydantic import BaseModel

from rag_config import rag, rag_2025
from agents import Agent, Runner, function_tool
from src.prompt import agent_system_prompt

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define tools
@function_tool
async def async_answer(query: str) -> str:
    """
    Answer queries strictly related to university admissions for general cases.
    """
    start = time.time()
    response = await rag.acontextual_rag_search(query, k=5)
    print("Response: ", response)
    print("Time (async_answer): ", time.time() - start)
    return response

@function_tool 
async def async_answer_2025(query: str) -> str:
    """
    Answer queries strictly related to university admissions in 2025.
    """
    start = time.time()
    response = await rag_2025.acontextual_rag_search(query, k=5)
    print("Response: ", response)
    print("Time (async_answer_2025): ", time.time() - start)
    return response

class Message(BaseModel):
    content: str

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "agent",
        Agent(
            name="UIT Chatbot",
            instructions=agent_system_prompt,
            tools=[async_answer, async_answer_2025],
            model="gpt-4o-mini",
        ),
    )

# Endpoint for chatbot interaction
@app.post("/query/")
async def run(query: Message):
    try:
        start_time = time.time()

        # Read user message from query parameter
        query = query.content

        print("User Query:", query)

        agent = cl.user_session.get("agent")

        result = Runner.run(agent, input=query)
        
        print("Result:", result)
        print("Total Time: ", time.time() - start_time)

        return JSONResponse(content={"result": result})
    
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# mount_chainlit(app=app, target="my_app.py", path="/chainlit")
if __name__ == "__main__":
    uvicorn.run("main:my_app", host="0.0.0.0", port=8000)
