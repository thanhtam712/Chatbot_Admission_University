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

# list_query = ["Học bổng?",
#               "Hoc phí?",
#               "Điều kiện xét tuyển?",
#               "Thời gian xét tuyển?",
#               "Thời gian nhập học?",
#               "điểm chuẩn của UIT là bao nhiêu?",
#               "Không biết lập trình thì liệu có thể theo học tại trường không ?",
#               "Không biết lập trình thì liệu có thể theo học tại trường ở ngành khác khmt được không ?",
#               "Các chính sách học bổng của trường ?",
#               "Học phí của Trường là bao nhiêu?",
#               "Môi trường học tập, đời sống sinh viên tại UIT như thế nào ?",
#               "Ngành khoa học máy tính đào tạo những gì?",
#               "Ngành trí tuệ nhân tạo đào tạo những gì ?",
#               "Ngành công nghệ thông tin của UIT đào tạo những gì ?",]

list_query = ["Tổ hợp môn xét tuyển năm 2025",]

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
    database = {"query": [], "contextual_rag": []}
    for q in list_query:
        result = asyncio.run(rag.acontextual_rag_search(q, debug=True, k=10))
        print(f"{GREEN}Query: {RESET}{q}")
        print(f"{GREEN}Contextual RAG: {RESET}{result}")
        
        # result_no_expand = rag.contextual_rag_search(q, debug=True, k=10, query_expand=False)
        # print(f"{GREEN}Contextual no expand query RAG: {RESET}{result_no_expand}") 


        input("Press Enter to continue...")
        
        
        
