# This file will request to api 192.168.20.150:8000/query to search query

import argparse
import requests
import asyncio
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel

GREEN = "\033[92m"
RESET = "\033[0m"

API_URL = "http://192.168.20.150:8777/query"

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

class Message(BaseModel):
    content: str

def search():
    
    args = parse_args()
    file_output = args.output
    questions = open("/mlcv2/WorkingSpace/Personal/hienht/uit_chatbot/add_data/dataset/cauhoi_tuyensinh.csv", "r").readlines()
    
    df_output = pd.DataFrame(columns=["Question", "Answer"])
    
    for q in tqdm(questions, desc="Processing questions", total=len(questions)):
        
        # RAG with es - 0.2 and se - 0.8
        query = Message(content=q.strip())
        result = requests.post(
            url=API_URL,
            json=query,
        )
        print(f"{GREEN}Contextual RAG Qdrant 0.5 - ES 0.5: {RESET}{result}")
        input("Press Enter to continue...")
        
        new_row = pd.DataFrame({
            "Question": [q],
            "Answer": [result],
        })
        
        df_output = pd.concat([df_output, new_row], ignore_index=True)
        
    print(f"Total questions: {len(questions)}")
    df_output.to_csv(file_output, index=False)
    
    
search()
   