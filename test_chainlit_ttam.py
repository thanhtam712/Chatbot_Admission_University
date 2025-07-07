import os
import time
from dotenv import load_dotenv
from chainlit.types import ThreadDict
from typing import Dict, Optional
from src.settings import Settings
from src.embedding import RAG
from llama_index.llms.openai import OpenAI
from llama_index.core.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent

import chainlit as cl


load_dotenv()

url_api = os.getenv("SYSTEM_API")

agent_system_prompt = "You are a supportive university admissions counselor chatbot. Please use answer tool as much as possible if user's query is related to any university admissions. Please use the reesponse from answer's tool to answer user's query, make sure to include any existed resources in the response. Add this statement in your response: 'Vui lòng liên hệ phòng tư vấn tuyển sinh của trường để biết thêm chi tiết nhé."

rag = RAG(setting=Settings())


def load_tool():

    def answer(query: str) -> str:
        """
        Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, etc.—while providing clear, empathetic guidance and assessing query urgency.
        """
        start = time.time()
        response = rag.contextual_rag_search(query)
        print("response_system: ", response)
        print("time retrieval system: ", time.time() - start)
        # prompt = f"Answer: {response["answer"]} \n\n Resources: {response["resources"]}"
        return response

    return FunctionTool.from_defaults(
        fn=answer,
        description="Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, etc.—while providing clear, empathetic guidance and assessing query urgency.",
        # return_direct=True,
    )


llm = OpenAI(
    "gpt-4o-mini",
    system_prompt="You are a supportive university admissions counselor chatbot. Please use answer tool as much as possible if user's query is related to any university admissions.",
)


@cl.on_chat_start
async def on_chat_start():
    chat_messages = []
    agent = OpenAIAgent.from_tools(
        tools=[load_tool()],
        verbose=True,
        system_prompt=agent_system_prompt,
        llm=llm,
    )
    cl.user_session.set("chat_messages", chat_messages)
    cl.user_session.set("agent", agent)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    new_memory = []
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]

    agent = OpenAIAgent.from_tools(
        tools=[load_tool()],
        verbose=True,
        system_prompt=agent_system_prompt,
        llm=llm,
    )

    # CODE HERE
    for message in root_messages:
        if message["type"] == "user_message":
            new_memory.append(ChatMessage(role="user", content=message["output"]))
        else:
            new_memory.append(ChatMessage(role="assistant", content=message["output"]))

    cl.user_session.set("chat_state", new_memory)
    cl.user_session.set("agent", agent)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Tuyển sinh ở UIT cần lưu ý những gì",
            message="Tuyển sinh ở UIT cần lưu ý những gì",
        ),
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


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    print(f"Default user: {default_user}")
    return default_user


@cl.on_message
async def run(message: cl.Message):
    try:
        start_time = time.time()

        chat_messages = cl.user_session.get("chat_messages")

        agent = cl.user_session.get("agent")

        result = await cl.make_async(agent.chat)(message.content, chat_messages)
        response = result.response

        chat_messages.append(ChatMessage(role="user", content=message.content))
        chat_messages.append(ChatMessage(role="assistant", content=response))

        cl.user_session.set("chat_messages", chat_messages)

        print(f"time retrieval system in app.py: {time.time() - start_time}")

        print(f"Result: {response}")
        print(f"Query: {message}")

        msg = cl.Message(content="", author="Assistant")
        final_response = ""

        for token in response:
            final_response += token + " "
            await msg.stream_token(token)

        await msg.send()

    except AssertionError as e:
        print("ERROR: AssertionError encountered:", e)
    except Exception as e:
        print("ERROR: Unexpected error:", e)
