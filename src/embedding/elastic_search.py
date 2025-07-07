from typing import Literal
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from llama_index.core.bridge.pydantic import Field

from src.utils import get_formatted_logger
from src.schemas import DocumentMetadata, ElasticSearchResponse, RAGType

logger = get_formatted_logger(__file__)


class ElasticSearch:
    """
    ElasticSearch client to index and search documents for contextual RAG.
    """

    url: str = Field(..., description="Elastic Search URL")

    def __init__(self, url: str, index_name: str):
        """
        Initialize the ElasticSearch client.

        Args:
            url (str): URL of the ElasticSearch server
            index_name (str): Name of the index used to be created for contextual RAG
        """
        self.es_client = Elasticsearch(
            url,
            request_timeout=30,
            max_retries=10,
            retry_on_timeout=True,
            verify_certs=False,
        )
        self.index_name = index_name
        self.create_index()

        logger.info(f"Connected to ElasticSearch at {url}")

    def create_index(self):
        """
        Create the index for contextual RAG from provided index name.
        """
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "text", "index": False},
                    "file_name": {"type": "text", "index": False},
                }
            },
        }

        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            logger.info(f"Created index {self.index_name}")

    def index_documents(
        self,
        documents_metadata: list[DocumentMetadata],
        type: Literal["origin", "contextual", "both"] = "original",
    ) -> bool:
        """
        Index the documents to the ElasticSearch index.

        Args:
            documents_metadata (list[DocumentMetadata]): List of documents metadata to index.
        """

        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "doc_id": metadata.doc_id,
                    "file_name": metadata.file_name,
                    "content": metadata.original_content,
                    "contextualized_content": metadata.contextualized_content,
                },
            }
            for metadata in documents_metadata
        ]

        success, _ = bulk(self.es_client, actions)
        if success:
            logger.info(f"Indexed {len(documents_metadata)} documents")

        self.es_client.indices.refresh(index=self.index_name)

        return success

    def search(self, query: str, k: int = 20) -> list[ElasticSearchResponse]:
        """
        Search the documents relevant to the query.

        Args:
            query (str): Query to search
            k (int): Number of documents to return

        Returns:
            list[ElasticSearchResponse]: List of ElasticSearch response objects.
        """
        logger.info(f"query: {query}")

        self.es_client.indices.refresh(
            index=self.index_name
        )  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                    # "fields": ["contextual_content"]
                }
            },
            "size": k,
        }
        
        response = self.es_client.search(index=self.index_name, body=search_body)

        return [
            ElasticSearchResponse(
                doc_id=hit["_source"]["doc_id"],
                file_name=hit["_source"]["file_name"],
                content=hit["_source"]["content"],
                contextualized_content=hit["_source"]["contextualized_content"],
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]
