from enum import Enum
from pydantic import Field
from llama_index.core.bridge.pydantic import BaseModel


    
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

class ParamTool(BaseModel):
    query: str

class RAGType:
    """
    RAG type schema.

    Attributes:
        ORIGIN (str): Origin RAG type.
        CONTEXTUAL (str): Contextual RAG type.
        BOTH (str): Both Origin and Contextual RAG type.
    """

    ORIGIN = "origin"
    CONTEXTUAL = "contextual"
    BOTH = "both"


class DocumentMetadata(BaseModel):
    """
    Document metadata schema.

    Attributes:
        doc_id (str): Document ID.
        original_content (str): Original content of the document.
        contextualized_content (str): Contextualized content of the document which will be prepend to the original content.
    """

    doc_id: str
    file_name: str
    original_content: str
    contextualized_content: str


class ElasticSearchResponse(BaseModel):
    """
    ElasticSearch response schema.

    Attributes:
        doc_id (str): Document ID.
        content (str): Content of the document.
        contextualized_content (str): Contextualized content of the document.
        score (float): Score of the document.
    """

    doc_id: str
    file_name: str
    content: str
    contextualized_content: str
    score: float


class LLMService(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    GEMINI = "gemini"


class EmbeddingService(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class RerankerService(str, Enum):
    COHERE = "cohere"
    RANKGPT = "rankgpt"


class LLMConfig(BaseModel):
    service: LLMService
    model: str


class EmbeddingConfig(BaseModel):
    service: EmbeddingService
    model: str


class RerankerConfig(BaseModel):
    service: RerankerService
    model: str | None


class SupportDependencies(BaseModel):
    """
    Support dependencies schema.

    Attributes:
        query (str): Query.
    """

    query: str


class SupportResult(BaseModel):
    """
    Support result schema.

    Attributes:
        response (str): Response.
        # check_topic (bool): Check topic user ask related to topic in database chatbot.
    """

    response: str
    # check_topic: bool


class QueryRequest(BaseModel):
    """
    Query request schema.

    Attributes:
        content (str): Content of the query.
    """

    content: str

class CallResponse(BaseModel):
    """
    Call response schema.

    Attributes:
        response (str): Response.
        resources (List[str]): Resources.
    """

    response: str = Field(description="Response to the query.")
    resources: list[str] = Field(description="Resources related to the query, such as links to web, file pdf, file image, etc.")
    



class ChatState(BaseModel):
    """State manager for chat history."""
    messages: List[ModelMessage] = Field(
        default_factory=list,
        description="Chat conversation history",
        max_length=1000  # Prevent memory issues
    )
    
    def append_messages(self, new_messages: List[ModelMessage]) -> None:
        """Add new messages to history."""
        self.messages.extend(new_messages)
        if len(self.messages) > 1000:
            # Keep most recent messages if we exceed the limit
            self.messages = self.messages[-1000:]
            
    def get_all_messages(self) -> List[ModelMessage]:
        """Get new messages since the last call."""
        return self.messages
    
    def clear_messages(self) -> None:
        """Reset chat history."""
        self.messages.clear()


class SystemOutput(BaseModel):
    """
    System output schema.

    Attributes:
        response (str): Response.
        resources (List[str]): Resources.
    """

    is_related_to_admission: bool = Field(description="Check if the user query is related to university admissions.")
    reason: str = Field(description="Reason why the user query is related to university admissions.")
