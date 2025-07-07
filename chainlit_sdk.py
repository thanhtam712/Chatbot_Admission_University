import time
import chainlit as cl
from dotenv import load_dotenv
from chainlit.types import ThreadDict
from typing import Dict, Optional

from flask import redirect, url_for, send_from_directory

from agents import Runner, Agent, function_tool
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.responses.response_input_item_param import Message

from rag_config import rag, rag_2025
from src.prompt import agent_system_prompt
from src.tools.abbreviate import get_abrreviate
from src.utils.logger import get_formatted_logger

load_dotenv()
logger = get_formatted_logger(__file__, "logs/rag_es.log")

@function_tool
async def answer(query: str) -> str:
    """
    Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency.
    """
    start = time.time()
    response = await rag.acontextual_rag_search(query, k=5)
    # print("response_system: ", response)
    print("async time retrieval system: ", time.time() - start)
    return response


@function_tool
async def answer_only_2025(query: str) -> str:
    """
    Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency in 2025.
    """
    start = time.time()
    response = await rag_2025.acontextual_rag_search(query, k=5)
    # print("response_system: ", response)
    print("async time retrieval system: ", time.time() - start)
    return response


@cl.on_chat_start
async def on_chat_start():
    chat_messages = []
    cl.user_session.set("chat_messages", chat_messages)

    abbreviations = get_abrreviate()
    cl.user_session.set("abbreviations", abbreviations)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    new_memory = []
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    # CODE HERE
    for message in root_messages:
        if message["output"]:
            if message["type"] == "user_message":
                new_memory.append(Message(role="user", content=message["output"]))
            else:
                new_memory.append(Message(role="assistant", content=message["output"]))

    cl.user_session.set("chat_messages", new_memory)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Địa chỉ trường UIT ở đâu?",
            message="Địa chỉ trường UIT ở đâu?",
        ),
        cl.Starter(
            label="Tại sao nên học ở UIT",
            message="Tại sao nên học ở UIT",
        ),
        cl.Starter(
            label="Năm 2025 trường có ngành gì mới?",
            message="Năm 2025 trường có ngành gì mới?",
        ),
    ]


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    if default_user.metadata["provider"] != "google":
        return None
    # if "uit.edu.vn" not in default_user.identifier:
    #     return None
    return default_user


@cl.on_message
async def run(message: cl.Message):
    try:
        start_time = time.time()

        chat_messages = cl.user_session.get("chat_messages")

        print(f"chat_messages: {chat_messages}")
        agent = Agent(
            name="UIT Chatbot",
            instructions=agent_system_prompt.format(
                abbreviations_str=cl.user_session.get("abbreviations")
            ),
            tools=[answer, answer_only_2025],
            model="gpt-4.1-mini",
        )

        expand_query = message.content
        # expand_query = await rag.apreprocess_query(message.content)

        print("expand_query: ", expand_query)
        print("message.content: ", message.content)

        input_conversation: list[Message] = [
            Message(role="user", content=c["content"]) for c in chat_messages[-10:]
        ]
        input_conversation.append(Message(role="user", content=expand_query))

        result = Runner.run_streamed(agent, input=input_conversation)
        response = ""
        msg = cl.Message(content="", author="Assistant")
        blabla = 0

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                if blabla == 0:
                    print(f"first token in app.py: {time.time() - start_time}")
                    blabla = 1

                print(event.data.delta, end="", flush=True)
                response += event.data.delta
                await msg.stream_token(event.data.delta)

        await msg.send()

        chat_messages.append(Message(role="user", content=message.content))
        chat_messages.append(Message(role="assistant", content=response))

        cl.user_session.set("chat_messages", chat_messages)

        print(f"time retrieval system in app.py: {time.time() - start_time}")

        print(f"Result: {response}")

    except AssertionError as e:
        print("ERROR: AssertionError encountered:", e)
    except Exception as e:
        print("ERROR: Unexpected error:", e)
