import time
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.responses.response_input_item_param import Message

from rag_config import llm
from src.tools.abbreviate import get_abrreviate
from src.prompt_manual import classify_query_prompt, check_history_prompt, QA_ANSWER_PROMPT, choose_tool_prompt
from src.utils.logger import get_formatted_logger
from src.settings import Settings
from src.embedding import RAG

setting_rag = Settings()
setting_rag.contextual_rag_collection_name = "contextual_2024"
setting_rag.elastic_search_index_name = "contextual_ori_2024"
rag = RAG(setting=setting_rag)
setting_rag_2025 = Settings()
setting_rag_2025.contextual_rag_collection_name = "contextual_2025"
setting_rag_2025.elastic_search_index_name = "contextual_ori_2025"
rag_2025 = RAG(setting=setting_rag_2025)

load_dotenv()
logger = get_formatted_logger(__file__, "logs/rag_es.log")

async def answer(query: str) -> str:
    """
    Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency.
    """
    start = time.time()
    response = await rag.acontextual_rag_search(query, k=5)
    print("async time retrieval system: ", time.time() - start)
    return response


async def answer_only_2025(query: str) -> str:
    """
    Answer queries strictly related to university admissions—covering application procedures, tuition, scholarships, majors, etc.—while providing clear, empathetic guidance and assessing query urgency in 2025.
    """
    start = time.time()
    response = await rag_2025.acontextual_rag_search(query, k=5)
    print("async time retrieval system: ", time.time() - start)
    return response

async def get_chat_classify(query: str) -> str:
    """
    Classify the query to determine if it is related to university admissions.
    """
    prompt_classify = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=classify_query_prompt.format(),
            ),
            ChatMessage(
                role="user",
                content="Xác định câu truy vấn sau có liên quan đến nội dung tư vấn tuyển sinh hoặc tìm hiểu về trường không? \n Input: {query}".format(
                    query=query,
                ),
            ),
        ]
    )
    
    messages = prompt_classify.format_messages()
    response = await llm.achat(messages)
    response = response.message.content
    
    return response

async def choose_tool(query: str) -> str:
    """
    Choose the appropriate tool based on the query.
    """
    prompt_choose_tool = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=choose_tool_prompt.format(),
            ),
            ChatMessage(
                role="user",
                content="Chọn công cụ nào để trả lời câu truy vấn này: \n Input: {query}".format(
                    query=query,
                ),
            ),
        ]
    )
    
    messages = prompt_choose_tool.format_messages(query=query)
    response = await llm.achat(messages)
    response = response.message.content
    
    return response

async def get_chat_check_history(query: str, messages_history: list[Message]) -> str:
    prompt_check_history = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=check_history_prompt.format(),
            ),
            ChatMessage(
                role="user",
                content="Kiểm tra nội dung trong lịch sử có liên quan đến câu hỏi này không? \n Input: Lịch sử chat của người dùng: {messages_history} \n Câu truy vấn: {query} ".format(
                    query=query,
                    messages_history=messages_history,
                ),
            ),
        ]
    )
    
    messages = prompt_check_history.format_messages(query=query)
    response = await llm.achat(messages)
    response = response.message.content
    
    return response

async def get_chat_answer(query: str, contexts_resources: str) -> str:
    prompt_answer = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=QA_ANSWER_PROMPT.format(),
            ),
            ChatMessage(
                role="user",
                content="Trả lời câu truy vấn này: \n Input: {query} \n Nội dung và nguồn tài liệu liên quan: {contexts_resources}".format(
                    query=query,
                    contexts_resources=contexts_resources,
                ),
            ),
        ]
    )
    
    messages = prompt_answer.format_messages(query=query)
    response = await llm.achat(messages)
    response = response.message.content
    
    return response

async def run(query: str):
    try:
        start_time = time.time()

        print("query from user: ", query)

        # Check if the query is related to 2025
        classify_query = await get_chat_classify(query)
        print("classify_query: ", classify_query)
        
        answer_tool = "Empty"
        if classify_query != "Đúng":
            result = classify_query
        
        else:
            print("Query related to Admissions")

            chat_messages = []
            query_expanded = await get_chat_check_history(query, chat_messages)

            tool = await choose_tool(query_expanded)
            print("tool: ", tool)
            
            if "answer" in tool:
                print("Using answer tool")
                answer_tool = await answer(query_expanded)
            elif "answer_only_2025" in tool:
                print("Using answer_only_2025 tool")
                answer_tool = await answer_only_2025(query_expanded) 
            else:
                print("Error, tool not found")
                answer_tool = "Empty"

            if answer_tool != "Empty":
                result = await get_chat_answer(query_expanded, answer_tool)
            else:
                print("Error, answer_tool is empty")
                result = "Empty"

            print("answer_tool: ", answer_tool)
            
        return result, answer_tool

    except AssertionError as e:
        print("ERROR: AssertionError encountered:", e)
    except Exception as e:
        print("ERROR: Unexpected error:", e)


async def main():
    dataset_path = Path("/mlcv2/WorkingSpace/Personal/hienht/uit_chatbot/add_data/dataset/DATASET_331.csv")
    dataset = pl.read_csv(dataset_path, truncate_ragged_lines=True)
    questions = dataset["Question"].to_list()
    answers = dataset["Answer_Agent"].to_list()
    answers_4LLM = dataset["Answer_4LLM"].to_list()
    
    judge_prompt = """
    Bạn là một chuyên gia tư vấn tuyển sinh của trường đại học Công nghệ thông tin - ĐHQG TP.HCM.
    ==============================
    Input:
    Với câu hỏi: <câu hỏi>
    Đáp án đúng của câu hỏi: <đáp án đúng> 
    Đây là đáp án từ hệ thống: <đáp án từ hệ thống>   
    
    Output:
    <Đúng hoặc Sai>
    =============================
    Hãy suy nghĩ và phân tích đáp án đúng của câu hỏi trên và đưa ra đánh giá về độ chính xác của đáp án từ hệ thống.
    Nếu bạn thấy đáp án từ hệ thống là chính xác, hãy trả lời "Đúng". Nếu bạn thấy đáp án từ hệ thống là không chính xác, hãy trả lời "Sai".
    Bạn chỉ cần trả lời "Đúng" hoặc "Sai" mà không cần giải thích thêm.
    
    Ví dụ:
    Input:
    Với câu hỏi:
    Địa chỉ trường UIT ở đâu?
    Đáp án đúng của câu hỏi: Khu phố 6, Phường Linh Trung, TP. Thủ Đức, Thành phố Hồ Chí Minh.
    Đây là đáp án từ hệ thống:
    Trường Đại học Công nghệ thông tin - ĐHQG TP.HCM, Khu phố 6, Phường Linh Trung, TP. Thủ Đức, Thành phố Hồ Chí Minh.
    
    Output:
    Đúng
    """
    
    user_prompt = """
    Với câu hỏi: {question} \nĐáp án đúng của câu hỏi: {answer}\nĐáp án từ hệ thống: {result}
    """
    
    output_csv = "add_data/judged/acc_contextual_2.csv"
    output_txt = "add_data/judged/acc_contextual_2.txt"
    f_txt = open(output_txt, "a+", encoding="utf-8")
    ques_list, gt_list, result_list, judge_list, context_list = [], [], [], [], []
    
    for ques, ans, ans_4llm in zip(questions, answers, answers_4LLM):
        print("ques: ", ques)
        # print("ans: ", ans)
        if ans_4llm is None or "Tôi không thể trả lời câu hỏi này" in ans_4llm:
            result, context = await run(ques)
        else:
            result = ans_4llm
            context = ""
            
        prompt_answer = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=judge_prompt.format(),
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt.format(question=ques, answer=ans, result=result),
                ),
            ]
        )

        messages = prompt_answer.format_messages()
        response = await llm.achat(messages)
        response = response.message.content
        
        ques_list.append(ques)
        gt_list.append(ans)
        context_list.append(context)
        result_list.append(result)
        judge_list.append(response)
        
        f_txt.write(f"Ques: {ques}\n")
        f_txt.write(f"Groundtruth: {ans}\n")
        f_txt.write(f"Result: {result}\n")
        f_txt.write(f"Judge: {response}\n")
        f_txt.write(f"Context: {context}\n")
        f_txt.write("===============================\n")
        
        print("Response:", response)
        
        time.sleep(1)
        # input("Press Enter to continue...")
    
    df_output = pl.DataFrame({"Ques": ques_list, "Groundtruth": gt_list, "result": result_list, "judge": judge_list, "context": context_list})    
    df_output.write_csv(output_csv)
    print("Output saved to: ", output_csv)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
