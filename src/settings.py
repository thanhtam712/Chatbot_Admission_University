import os
from dotenv import load_dotenv
from mmengine.config import Config
from llama_index.core.bridge.pydantic import Field, BaseModel

from src.schemas import LLMConfig, EmbeddingConfig, RerankerConfig

load_dotenv(dotenv_path="./uit_chatbot/.env.production", override=True)

config = Config.fromfile("config.py")


class Settings(BaseModel):
    """
    Settings for the contextual RAG.

    Attributes:
        chunk_size (int): Default chunk size
        service (str): The LLM service, e.g., "openai"
        model (str): The LLM model name, e.g., "gpt-4o-mini"
        original_rag_collection_name (str): The original RAG collection name
        contextual_rag_collection_name (str): The contextual RAG collection name
        qdrant_host (str): The Qdrant host
        qdrant_port (int): The Qdrant port
        elastic_search_url (str): The ElasticSearch URL
        elastic_search_index_name (str): The ElasticSearch index name
        num_chunks_to_recall (int): The number of chunks to recall
        semantic_weight (float): The semantic weight
        bm25_weight (float): The BM25 weight
        top_n (int): Top n documents after reranking
    """

    chunk_size: int = Field(description="The chunk size", default=1024)

    llm_config: LLMConfig = Field(
        description="The LLM config",
        default=LLMConfig(
            service=config.llm_config.service, model=config.llm_config.model
        ),
    )

    embedding_config: EmbeddingConfig = Field(
        description="The embedding config",
        default=EmbeddingConfig(
            service=config.embedding_config.service, model=config.embedding_config.model
        ),
    )

    reranker_config: RerankerConfig = Field(
        description="The reranker config",
        default=RerankerConfig(
            service=config.reranker_config.service,
            model=config.reranker_config.model,
        ),
    )
    
    original_rag_collection_name: str = Field(
        description="The original RAG collection name", default="ori_data_no_contextual"
    )
    
    contextual_rag_collection_name: str = Field(
        description="The contextual RAG collection name",
        default="contextual_2024",
    )

    qdrant_url: str = Field(
        description="The Qdrant URL", default=os.getenv("QDRANT_URL")
    )

    elastic_search_url: str = Field(
        description="The Elastic URL", default=os.getenv("ELASTIC_SEARCH_URL")
    )
    
    elastic_search_index_name: str = Field(
        description="The Elastic index name", default="contextual_ori_2024"
    )

    num_chunks_to_recall: int = Field(
        description="The number of chunks to recall", default=150
    )

    num_token_split_docx: int = Field(
        description="The number of token to split docx file", default=1500
    )

    # Reference: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
    semantic_weight: float = Field(description="The semantic weight", default=1)
    bm25_weight: float = Field(description="The BM25 weight", default=0)

    top_n: int = Field(description="Top n documents after reranking", default=2)
    
    collection_data: str = Field(description="Collection data", default="collection_data")


setting = Settings()
