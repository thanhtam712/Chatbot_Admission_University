import asyncio
import argparse
import threading
import pandas as pd
from src.embedding import RAG
from src.settings import Settings

GREEN = "\033[92m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(description="Demo for Contextual RAG")
parser.add_argument(
    "--q",
    type=str,
    help="Query",
    required=False,
)
parser.add_argument(
    "--compare",
    action="store_true",
    help="Compare the original RAG and the contextual RAG",
)

args = parser.parse_args()

setting = Settings()
setting.contextual_rag_collection_name = "contextual_2025"
setting.elastic_search_index_name = "contextual_ori_2025"
rag = RAG(setting)

if args.compare:
    thread = [
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Origin RAG: {RESET}{rag.origin_rag_search(q)}"
            )
        ),
        threading.Thread(
            target=lambda: print(
                f"\n\n{GREEN}Contextual RAG: {RESET}{rag.contextual_rag_search(q, debug=True)}"
            )
        ),
    ]

    for t in thread:
        t.start()

    for t in thread:
        t.join()
else:
    q = args.q
    database = {"query": [], "contextual_rag": []}
    result = asyncio.run(rag.acontextual_rag_search(q, debug=True, k=10))
    print(f"{GREEN}Query: {RESET}{q}")
    print(f"{GREEN}Contextual RAG: {RESET}{result}")
        
