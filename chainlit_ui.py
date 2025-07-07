import time
import chainlit as cl
from typing import Dict, Optional
from chainlit.types import ThreadDict
from pydantic_ai import Agent, Tool

from src.schemas import SupportResult, ChatState
from src.embedding.rag_es import RAG
from src.settings import setting


@cl.on_chat_start
async def start():
    rag = RAG(setting)
    cl.user_session.set("rag", rag)
    chat_state = ChatState()
    cl.user_session.set("chat_state", chat_state)


async def response_system(query: str):
    """
    Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, etc.—while providing clear, empathetic guidance and assessing query urgency.
    """
    start = time.time()
    rag = cl.user_session.get("rag")
    res = rag.contextual_rag_search(query, debug=True)
    print(f"Time to retrieve response in system: {time.time() - start}")
    return SupportResult(response=res)


support_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ChatState,
    result_type=SupportResult,
    system_prompt=("You are a supportive university admissions counselor chatbot. "),
    tools=[
        Tool(
            response_system,
            description="Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, etc.—while providing clear, empathetic guidance and assessing query urgency. Include links to official university resources or other relevant sources for further information and ensure that all links provided are directly related to the query. Do not include unrelated links.",
        ),
    ],
)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    new_memory = ChatState()
    new_memory.messages = []
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]

    # CODE HERE
    for message in root_messages:
        if message["type"] == "user_message":
            new_memory.append_messages(
                [cl.ChatMessage(role="user", content=message["output"])]
            )
        else:
            new_memory.append_messages(
                [cl.ChatMessage(role="assistant", content=message["output"])]
            )

    cl.user_session.set("chat_state", new_memory)


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Học phí ở UIT",
            message="Học phí ở UIT",
        ),
        cl.Starter(
            label="Học bổng ở UIT",
            message="Học bổng ở UIT",
        ),
        cl.Starter(
            label="Quy trình xét tuyển ở UIT",
            message="Quy trình xét tuyển ở UIT",
        ),
    ]


@cl.on_message
async def run(message: cl.Message):
    try:
        # chat_state = cl.user_session.get("chat_state")

        # all_messages = chat_state.get_all_messages()
        chat_state = cl.user_session.get("chat_state")
        all_messages = chat_state.get_all_messages()
        print(f"Chat state history: {all_messages}")

        start_time = time.time()

        result = support_agent.run_sync(
            message.content, deps=chat_state, message_history=all_messages
        )

        print(f"time retrieval system in app.py: {time.time() - start_time}")

        chat_state.append_messages(result.new_messages())

        print(f"Result: {result.data}")
        query = result.data.response
        print(f"Query: {query}")

        msg = cl.Message(content="", author="Assistant")
        response = ""

        for token in query:
            response += token + " "
            await msg.stream_token(token)

        await msg.send()

    except AssertionError as e:
        print("ERROR: AssertionError encountered:", e)
    except Exception as e:
        print("ERROR: Unexpected error:", e)
