import bm25s
from pathlib import Path

from src.schemas import DocumentMetadata, ElasticSearchResponse
from src.utils import get_formatted_logger

logger = get_formatted_logger(__file__)


class BM25sSearch:
    """
    BM25s Search client to index and search documents for contextual RAG.
    """

    def __init__(self):
        """
        Initialize the BM25s Search client.
        """
        self.retriever = bm25s.BM25()
        self.stemmer_fn = lambda lst: [word for word in lst]

    def index_documents(self, documents_metadata: list[DocumentMetadata]) -> bool:
        """
        Index the documents to the BM25sSearch index.

        Args:
            documents_metadata (list[DocumentMetadata]): List of documents metadata to index.
        """
        corpus_token = []
        documents_metadata = [doc.contextualized_content for doc in documents_metadata]
        corpus_token = bm25s.tokenize(documents_metadata, stemmer=self.stemmer_fn)

        self.retriever.index(corpus_token)
        print("retriever bm25s", self.retriever)
        self.retriever.save("./bm25s_index", corpus=documents_metadata)

    def search(self, query: str, k: int = 20) -> list[ElasticSearchResponse]:
        """
        Search the documents relevant to the query.

        Args:
            query (str): Query to search
            k (int): Number of documents to return

        Returns:
            list[ElasticSearchResponse]: List of ElasticSearch response objects.
        """
        if Path("bm25s_index").exists():
            self.retriever.load("./bm25s_index", load_corpus=True)
        
        logger.info(f"query: {query}")

        query_token_ids = bm25s.tokenize(query, stemmer=self.stemmer_fn)
        print("retrieve", self.retriever)
        
        results, scores = self.retriever.retrieve(
            query_token_ids, k=k, show_progress=False
        )
        
        print(results)
        input()

        return [
            ElasticSearchResponse(
                doc_id=hit["_source"]["doc_id"],
                content=hit["_source"]["content"],
                contextualized_content=hit["_source"]["contextualized_content"],
                score=hit["_score"],
            )
            for hit in results
        ]
