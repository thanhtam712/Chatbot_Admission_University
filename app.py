import time
import enum
import uvicorn


from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from src.embedding import RAG
from src.settings import Settings
from src.schemas import QueryRequest
from src.utils.logger import get_formatted_logger

from pydantic import BaseModel


class AnswerType(str, enum.Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class Answer(BaseModel):
    answer: AnswerType  # câu trả lời yes/no/uncertain
    premises: list[str]  # những premises liên quan nhất đến câu trả lời
    reasoning_path: str  # Luồng reasoning để ra được kết quả


class Format(BaseModel):
    premises: list[str]  # những premises được cho sẵn
    question: str
    ground_truth: Answer


logger = get_formatted_logger(__name__, "logs/app.log")
app = FastAPI()
router = APIRouter()

setting = Settings()
rag = RAG(setting)


@router.get("/")
async def get_root() -> str:
    return JSONResponse(content={"message": "Hello World"})


@router.post("/query")
async def get_query(query_request: QueryRequest) -> str:
    logger.info(f"query_request: {query_request}")
    start_time = time.time()
    res = rag.acontextual_rag_search(query_request.content, debug=True)
    logger.info(f"result: {res}")
    logger.info(f"time retrieval system in app.py: {time.time() - start_time}")
    return JSONResponse(content={"result": res})


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
