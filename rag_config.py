import os
import time
from dotenv import load_dotenv
from agents import (
    Runner,
    Agent,
    function_tool,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,)
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool


from src.embedding import RAG
from src.settings import Settings
from src.schemas import SystemOutput

setting_rag = Settings()
setting_rag.contextual_rag_collection_name = "contextual_2024"
setting_rag.elastic_search_index_name = "contextual_ori_2024"
rag = RAG(setting=setting_rag)

setting_rag_2025 = Settings()
setting_rag_2025.contextual_rag_collection_name = "contextual_2025"
setting_rag_2025.elastic_search_index_name = "contextual_ori_2025"
rag_2025 = RAG(setting=setting_rag_2025)

load_dotenv()    

def load_tool():
    
    async def async_answer(query: str) -> str:
        """
        Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency another year like 2025.
        """
        start = time.time()
        response = await rag.acontextual_rag_search(query, k=5)
        
        print("response_system: ", response)
        print("async time retrieval system: ", time.time() - start)
        
        return response

    return FunctionTool.from_defaults(
        async_fn=async_answer,
        description="Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency another year like 2025.",
    )


def load_tool_2025():
    
    async def async_answer_2025(query: str) -> str:
        """
        Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency only when mentioned 2025 year.
        """
        start = time.time()
        response = await rag_2025.acontextual_rag_search(query, k=5)
        
        print("response_system: ", response)
        print("async time retrieval system: ", time.time() - start)
        
        return response

    return FunctionTool.from_defaults(
        async_fn=async_answer_2025,
        description="Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency only when mentioned 2025 year.",
    )

llm = OpenAI(
    "gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    system_prompt="You are a supportive Information Technology university admissions counselor chatbot. Please use answer tool as much as possible if user's query is related to any university admissions and informations about university.",
)

guardrail_agent = Agent(
    name="guardrail_agent",
    instructions="Check if the user query is related to university admissions, such as: study programs, application documents, admission regulations, types of examinations, enrollment quotas, tuition fees, admission scores (reference point in national high school exam), application guidelines, academic programs at the university, english language requirements, scholarships, admission methods, dual-degree programs, informations about major in Information Technology domain (major code, define major, introduce major,...), introduce about university (information about UIT, hotline, address, dorm, infrastructure, learning environment, student life, club, team, ...), priority admission according to National University regulations, etc.",
    output_type=SystemOutput,
)
