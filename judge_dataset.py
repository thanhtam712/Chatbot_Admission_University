import os
import time
import asyncio
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Optional
from dotenv import load_dotenv

from src.embedding.rag_ttam import RAG as RAG_TTAM
from src.embedding import RAG
from src.prompt import agent_system_prompt
from src.tools.abbreviate import get_abrreviate
from src.settings import setting, Settings

from agents import (
    Runner,
    Agent,
    function_tool,
)

from llama_index.llms.openai import OpenAI
from llama_index.core import ChatPromptTemplate

from rag_config import rag_2025, rag

load_dotenv()

GREEN = "\033[92m"
RESET = "\033[0m"

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


def parse_args():
    """
    Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser(description="Demo for Contextual RAG")
    parser.add_argument(
        "--file",
        type=str,
        help="File csv dataset",
        required=True,
    )
    parser.add_argument("--output", type=str, help="Output file", required=True)
    parser.add_argument("--search", action="store_true", help="Task judge")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the original RAG and the contextual RAG",
    )

    args = parser.parse_args()
    return args

def load_llm():
    """
    Load the LLM model - OpenAI.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(model="gpt-4o-mini", max_tokens=512, openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that evaluates whether the system's response is correct based on the given ground truth."),
        ("user", "Question: {question}. System response: {response}. Ground truth: {groundtruth}. Is the response correct? Answer only Yes or No.")
    ])
    
    return llm, prompt

def load_dataset():
    """
    Load dataset from csv file.
    """
    
    args = parse_args()
    
    df = pd.read_csv(args.file)
    questions = df["Question"].tolist()
    # groundtruths = df["Answer Agent"].tolist()
    groundtruths = []
    # status = df["Test"].tolist()[:215]
    # status = df["Eval"].tolist()

    file_output = args.output

    return questions, groundtruths, file_output


def load_rag():
    """
    Load RAG model with different settings.
    
    - semantic_weight = 0.5, bm25_weight = 0.5
    - semantic_weight = 0, bm25_weight = 1
    - semantic_weight = 1, bm25_weight = 0
    - semantic_weight = 0.8, bm25_weight = 0.2
    
    Returns:
        rag (RAG): RAG model with different settings.
        rag_es (RAG): RAG model with ES setting.
        rag_qdrant (RAG): RAG model with Qdrant setting.
        rag_qdrant_es (RAG): RAG model with Qdrant ES setting.
    """
    
    rag = RAG(setting)

    setting.semantic_weight = 0
    setting.bm25_weight = 1

    rag_es = RAG(setting)

    setting.semantic_weight = 1
    setting.bm25_weight = 0

    rag_qdrant = RAG(setting)

    setting.semantic_weight = 0.8
    setting.bm25_weight = 0.2

    rag_qdrant_es = RAG(setting)
    
    return rag, rag_es, rag_qdrant, rag_qdrant_es

def load_rag_ttam():
    """
    Load RAG model with different settings for TTAM. Search into 2 database: contextual and original.
    """
    rag_ttam = RAG_TTAM(setting)
    
    return rag_ttam
    
    

async def main():
    
    setting_contextual = Settings()
    setting_contextual.contextual_rag_collection_name = "contextual_2024"
    setting_contextual.elastic_search_index_name = "contextual_ori_2024"
    rag_contextual = RAG(setting_contextual)
    
    setting_not_contextual = Settings()
    setting_not_contextual.contextual_rag_collection_name = "chunk_both"
    setting_not_contextual.elastic_search_index_name = "chunk_both"
    rag_not_contextual = RAG(setting_not_contextual)
    
    questions, groundtruths, file_output = load_dataset()
    
    df_output = pd.DataFrame(columns=["Question", "Contextual_retrieved", "Not_Contextual_RAG"])
    
    count_ques = 0
    for q in tqdm(questions, desc="Processing questions", total=len(questions)):
        count_ques += 1
        # gt = groundtruths[questions.index(q)]
        
        # RAG with es - 0.5 and se - 0.5
        result_contextual = await rag_contextual.acontextual_rag_search(q, debug=True)
        print(f"{GREEN}Query: {RESET}{q}")
        print(f"{GREEN}Contextual RAG: {RESET}{result_contextual}")

        result_not_contextual = await rag_not_contextual.acontextual_rag_search(q, debug=True)
        print(f"{GREEN}Not Contextual RAG: {RESET}{result_not_contextual}")
        
        new_row = pd.DataFrame({
            "Question": [q],
            "Contextual_retrieved": [result_contextual],
            "Not_Contextual_RAG": [result_not_contextual]
        })
        
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
    print(f"Total questions: {len(questions)}")
    df_output.to_csv(file_output, index=False)
    

def search():
    args = parse_args()
    file_output = args.output
    questions = open("/mlcv2/WorkingSpace/Personal/hienht/uit_chatbot/add_data/dataset/cauhoi_tuyensinh.txt", "r").readlines()
    questions = set(list(map(lambda x: x.strip(), questions)))
    
    abbreviations = get_abrreviate()
    agent = Agent(
        name="UIT Chatbot",
        instructions=agent_system_prompt.format(
            abbreviations_str=abbreviations
        ),
        tools=[answer, answer_only_2025],
        model="gpt-4o-mini",
    )
    
    df_output = pd.DataFrame(columns=["Question", "Answer_es0.2_se0.8"])
    
    for q in tqdm(questions, desc="Processing questions", total=len(questions)):
        
        # RAG with es - 0.5 and se - 0.5
        # result =  asyncio.run(rag.acontextual_rag_search(q, debug=True))
        # print(f"{GREEN}Contextual RAG Qdrant 0.5 - ES 0.5: {RESET}{result}")
        
        # # RAG with es - 0 and se - 1
        # result_qdrant =  asyncio.run(rag_qdrant.acontextual_rag_search(q, debug=True))
        # print(f"{GREEN}Contextual RAG Qdrant: {RESET}{result_qdrant}")
        
        # RAG with es - 0.2 and se - 0.8
        # result_qdrant_es = asyncio.run(rag_qdrant_es.acontextual_rag_search(q, debug=True, k=5))
        result = Runner.run_sync(agent, input=q)
        print(f"{GREEN}Contextual RAG Qdrant ES: {RESET}{result}")
        
        new_row = pd.DataFrame({
            "Question": [q],
            "Answer_es0.2_se0.8": [result],
        })
        
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
    print(f"Total questions: {len(questions)}")
    df_output.to_csv(file_output, index=False)
    
    
    
def search_2_db():
    """
    Search into 2 database: contextual and original.
    """
    
    rag_ttam = load_rag_ttam()
    llm, prompt = load_llm()
    questions, groundtruths, status, file_output = load_dataset()
    
    df_output = pd.DataFrame(columns=["Question", "Groundtruth", "contextual_rag", "judge_1", "original_rag", "judge_2"])
    
    count_rag_ttam, count_rag_ttam_es, count_ques = 0, 0, 0
    
    for q in tqdm(questions, desc="Processing questions", total=len(questions)):
        
        if status[questions.index(q)] == "Bad":
            print(f"{GREEN}Query: {RESET}{q}")
            print(f"{GREEN}Status: {RESET}{status[questions.index(q)]}")
            continue 
        
        count_ques += 1
        gt = groundtruths[questions.index(q)]
        
        # RAG with es - 0.5 and se - 0.5
        result_contextual, result_original, _, _ = rag_ttam.contextual_rag_search(q, debug=True, query_expand=False)
        print(f"{GREEN}Query: {RESET}{q}")
        print(f"{GREEN}Contextual RAG: {RESET}{result_contextual}")
        print(f"{GREEN}Original RAG: {RESET}{result_original}")

        message = prompt.format_messages(question=q, response=result_contextual, groundtruth=gt)
        
        judge_result_contextual = llm.chat(message)
        if judge_result_contextual == "Yes":
            count_rag_ttam += 1
        print(f"{GREEN}Judge result: {RESET}{judge_result_contextual}")
        
        
        message = prompt.format_messages(question=q, response=result_original, groundtruth=gt)
        
        judge_result_original = llm.chat(message)
        if judge_result_original == "Yes":
            count_rag_ttam += 1
        print(f"{GREEN}Judge result: {RESET}{judge_result_original}")

        
        new_row = pd.DataFrame({
            "Question": [q],
            "Groundtruth": [groundtruths[questions.index(q)]],
            "contextual_rag": [result_contextual],
            "judge_1": [judge_result_contextual],
            "original_rag": [result_original],
            "judge_2": [judge_result_original]
        })
        
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
        # input("Press Enter to continue...")
        
    print(f"Accuracy RAG TTAM: {count_rag_ttam}")
    print(f"Accuracy RAG TTAM ES: {count_rag_ttam_es}")
    print(f"Total questions: {len(questions)}")
        
    df_output.to_csv(file_output, index=False)
    
def mrr_benchmark():
    """
    Calculate Mean Reciprocal Rank (MRR) for the dataset.
    """
    rag_ttam = load_rag_ttam()
    llm, prompt = load_llm()
    questions, groundtruths, status, file_output = load_dataset()
    
    df_output = pd.DataFrame(columns=["Question", "Groundtruth", "contextual_rag", "judge_1", "original_rag", "judge_2"])
    
    count_rag_ttam, count_rag_ttam_es, count_ques = 0, 0, 0
    
    for q in tqdm(questions[:50], desc="Processing questions", total=len(questions)):
        
        if status[questions.index(q)] == "Bad":
            print(f"{GREEN}Query: {RESET}{q}")
            print(f"{GREEN}Status: {RESET}{status[questions.index(q)]}")
            continue 
        
        count_ques += 1
        gt = groundtruths[questions.index(q)]
        
        # RAG with es - 0.5 and se - 0.5
        result_contextual, result_original, context_contextual, context_original = rag_ttam.contextual_rag_search(q, debug=True, query_expand=False)
        print(f"{GREEN}Query: {RESET}{q}")
        print(f"{GREEN}Contextual RAG: {RESET}{result_contextual}")
        print(f"{GREEN}Original RAG: {RESET}{result_original}")

        message = prompt.format_messages(question=q, response=result_contextual, groundtruth=gt)
        
        judge_result_contextual = llm.chat(message)
        if judge_result_contextual == "Yes":
            count_rag_ttam += 1
        print(f"{GREEN}Judge result: {RESET}{judge_result_contextual}")
        
        
        message = prompt.format_messages(question=q, response=result_original, groundtruth=gt)
        
        judge_result_original = llm.chat(message)
        if judge_result_original == "Yes":
            count_rag_ttam += 1
        print(f"{GREEN}Judge result: {RESET}{judge_result_original}")

        
        new_row = pd.DataFrame({
            "Question": [q],
            "Groundtruth": [groundtruths[questions.index(q)]],
            "contextual_rag": [result_contextual],
            "context_contextual": [context_contextual],
            "judge_1": [judge_result_contextual],
            "original_rag": [result_original],
            "context_original": [context_original],
            "judge_2": [judge_result_original]
        })
        
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
        print(f"Contextual RAG: {context_contextual}")
        print(f"Original RAG: {context_original}")
        print(f"Judge result contextual: {judge_result_contextual}")
        print(f"Judge result original: {judge_result_original}")
        
        input("Press Enter to continue...")
        
    print(f"Accuracy RAG TTAM: {count_rag_ttam}")
    print(f"Accuracy RAG TTAM ES: {count_rag_ttam_es}")
    print(f"Total questions: {len(questions)}")
        
    df_output.to_csv(file_output, index=False)
    
    
if __name__ == "__main__":    
    args = parse_args()

    # if args.search:
    #     search_2_db()
    # else:
    #     main()

    # mrr_benchmark()
    # search_2_db()
    # search()
    asyncio.run(main())
