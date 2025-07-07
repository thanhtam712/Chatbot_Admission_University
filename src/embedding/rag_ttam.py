import os
import sys
import uuid
import time
import asyncio
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms.function_calling import FunctionCallingLLM

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import ChatPromptTemplate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Settings,
    Document,
    StorageContext,
    VectorStoreIndex,
)

from src.prompt import (
    CONTEXTUAL_PROMPT,
    QA_PROMPT,
    CHUNK_AND_RESOURCE,
    EXPAND_QUERY_PROMPT,
    SYSTEM_QA_PROMPT,
    contextual_compression_prompt_template,
    contextual_compression_system_prompt_template,
)
from src.utils import get_formatted_logger
from src.tools import get_abrreviate, get_title_private
from src.embedding.elastic_search import ElasticSearch
from src.settings import Settings as ConfigSettings, setting as config_setting
from src.schemas import (
    RAGType,
    DocumentMetadata,
    LLMService,
    EmbeddingService,
    RerankerService,
)
from src.readers.paper_reader import llama_parse_read_paper, llama_read_txt_file


def time_format():
    now = datetime.now()
    return f'DEBUG - {now.strftime("%H:%M:%S")} - '


load_dotenv(
    dotenv_path="./uit_chatbot/.env.production",
    override=True,
)
es_api = os.getenv("ELASTIC_SEARCH_URL")
qdrant_api = os.getenv("QDRANT_URL")
print(f"Qdrant API: {qdrant_api}")
print(f"ElasticSearch API: {es_api}")


logger = get_formatted_logger(__file__)

Settings.chunk_size = config_setting.chunk_size


class RAG:
    """
    Retrieve and Generate (RAG) class to handle the indexing and searching of both Origin and Contextual RAG.
    """

    setting: ConfigSettings
    llm: FunctionCallingLLM
    splitter: SemanticSplitterNodeParser
    es: ElasticSearch
    qdrant_client: QdrantClient
    reranker: BaseNodePostprocessor

    def __init__(self, setting: ConfigSettings):
        """
        Initialize the RAG class with the provided settings.

        Args:
            setting (ConfigSettings): The settings for the RAG.
        """
        self.setting = setting

        embed_model = self.load_embedding(
            setting.embedding_config.service, setting.embedding_config.model
        )
        Settings.embed_model = embed_model

        self.llm = self.load_llm(setting.llm_config.service, setting.llm_config.model)
        Settings.llm = self.llm

        self.contextual_compressor_llm = self.llm

        # self.reranker = self.load_reranker(
        #     setting.reranker_config.service, setting.reranker_config.model
        # )

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        self.qdrant_client = QdrantClient(
            url=setting.qdrant_url, port=None, timeout=1000
        )

        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )

    def load_embedding(self, service: EmbeddingService, model: str) -> BaseEmbedding:
        """
        Load the embedding model.

        Args:
            service (EmbeddingService): The embedding service.
            model (str): The embedding model name.
        """
        logger.info("load_embedding: %s, %s", service, model)

        if service == EmbeddingService.HUGGINGFACE:
            return HuggingFaceEmbedding(model_name=model, cache_folder="models")

        elif service == EmbeddingService.OPENAI:
            return OpenAIEmbedding(model=model)

        else:
            raise ValueError("Service not supported.")

    def load_llm(self, service: LLMService, model: str) -> FunctionCallingLLM:
        """
        Load the LLM model.

        Args:
            service (LLMService): The LLM service.
            model (str): The LLM model name.
        """
        logger.info("load_llm: %s, %s", service, model)

        if service == LLMService.OPENAI:
            # return OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5, top_p=0.5, max_tokens=500)
            return OpenAI(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                top_p=0.2,
                max_tokens=1000,
            )

        elif service == LLMService.GROQ:
            return Groq(model=model, api_key=os.getenv("GROQ_API_KEY"))
        elif service == LLMService.GEMINI:
            return Gemini(model=model, api_key=os.getenv("GEMINI_API_KEY"))

        else:
            raise ValueError("Service not supported.")

    def load_reranker(self, service: str, model: str) -> BaseNodePostprocessor:
        """
        Load the reranker model.

        Args:
            service (str): The reranker service.
            model (str): The reranker model name. Default to `""`.
        """
        logger.info("load_reranker: %s, %s", service, model)

        if service == RerankerService.COHERE:
            return CohereRerank(
                top_n=self.setting.top_n, api_key=os.getenv("COHERE_API_KEY")
            )
        elif service == RerankerService.RANKGPT:
            llm = OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
            return RankGPTRerank(top_n=self.setting.top_n, llm=llm)
        else:
            raise ValueError("Service not supported.")

    def split_document(
        self,
        document: Document | list[Document],
        show_progress: bool = True,
    ) -> list[list[Document]]:
        """
        Split the document into chunks.

        Args:
            document (Document | list[Document]): The document to split.
            show_progress (bool): Show the progress bar.

        Returns:
            list[list[Document]]: List of documents after splitting.
        """
        assert isinstance(document, list)

        documents: list[list[Document]] = []

        document = tqdm(document, desc="Splitting...") if show_progress else document

        for doc in document:
            nodes = self.splitter.get_nodes_from_documents([doc])
            documents.append(
                [
                    Document(text=node.get_content(), metadata=node.metadata)
                    for node in nodes
                ]
            )

        return documents

    def add_contextual_content(
        self,
        origin_document: Document,
        splited_documents: list[Document],
    ) -> tuple[list[Document], list[DocumentMetadata]]:
        """
        Add contextual content to the splited documents.

        Args:
            origin_document (Document): The original document.
            splited_documents (list[Document]): The splited documents from the original document.

        Returns:
            (tuple[list[Document], list[DocumentMetadata]]): List of documents with contextual content and its metadata.
        """

        whole_document = origin_document.text
        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        for chunk in splited_documents:
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                ChatMessage(
                    role="user",
                    content=CONTEXTUAL_PROMPT.format(
                        WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                    ),
                ),
            ]

            response = self.llm.chat(messages)
            contextualized_content = response.message.content

            # Prepend the contextualized content to the chunk
            new_chunk = contextualized_content + "\n\n" + chunk.text

            # Manually generate a doc_id for indexing in elastic search
            doc_id = str(uuid.uuid4())
            documents.append(
                Document(
                    text=new_chunk,
                    metadata={
                        "doc_id": doc_id,
                        "file_name": chunk.metadata["file_name"],
                        **chunk.metadata,
                    },
                ),
            )
            documents_metadata.append(
                DocumentMetadata(
                    doc_id=doc_id,
                    file_name=chunk.metadata["file_name"],
                    original_content=whole_document,
                    contextualized_content=contextualized_content,
                ),
            )

        return documents, documents_metadata

    def convert_format_ingest_data(
        self, raw_documents: list[Document]
    ) -> list[Document]:
        """
        Convert the format of the raw documents to the ingest documents.

        Args:
            raw_documents (list[Document]): List of raw documents.

        Returns:
            list[Document]: List of ingest documents.
        """
        ingest_documents = []
        doc_id = str(uuid.uuid4())

        for doc in raw_documents:
            ingest_documents.append(
                DocumentMetadata(
                    doc_id=doc_id,
                    file_name=doc.metadata["file_name"],
                    original_content=doc.text,
                    contextualized_content="",
                )
            )
        return ingest_documents

    def get_contextual_documents(
        self, raw_documents: list[Document], splited_documents: list[list[Document]]
    ) -> tuple[list[Document], list[DocumentMetadata]]:
        """
        Get the contextual documents from the raw and splited documents.

        Args:
            raw_documents (list[Document]): List of raw documents.
            splited_documents (list[list[Document]]): List of splited documents from the raw documents one by one.

        Returns:
            (tuple[list[Document], list[DocumentMetadata]]): Tuple of contextual documents and its metadata one by one.
        """

        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        assert len(raw_documents) == len(splited_documents)

        for raw_document, splited_document in tqdm(
            zip(raw_documents, splited_documents),
            desc="Adding contextual content ...",
            total=len(raw_documents),
        ):
            document, metadata = self.add_contextual_content(
                raw_document, splited_document
            )
            documents.extend(document)
            documents_metadata.extend(metadata)

        return documents, documents_metadata

    def ingest_data(
        self,
        documents: list[Document],
        show_progress: bool = True,
        collection_name: str = "",
        type: Literal["origin", "contextual"] = "contextual",
    ):
        """
        Ingest the data to the QdrantVectorStore.

        Args:
            documents (list[Document]): List of documents to ingest.
            show_progress (bool): Show the progress bar.
            type (Literal["origin", "contextual"]): The type of RAG to ingest.
        """

        # if type == "origin":
        #     collection_name = self.setting.original_rag_collection_name
        # else:
        #     collection_name = self.setting.contextual_rag_collection_name

        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=show_progress
        )

        return index  # noqa

    def insert_data(
        self,
        documents: list[Document],
        show_progess: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):
        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name

        vector_store_index = self.get_qdrant_vector_store_index(
            client=self.qdrant_client,
            collection_name=collection_name,
        )

        documents = (
            tqdm(documents, desc=f"Adding more data to {type} ...")
            if show_progess
            else documents
        )
        for document in documents:
            vector_store_index.insert(document)

    def get_qdrant_vector_store_index(
        self, client: QdrantClient, collection_name: str
    ) -> VectorStoreIndex:
        """
        Get the QdrantVectorStoreIndex from the QdrantVectorStore.

        Args:
            client (QdrantClient): The Qdrant client.
            collection_name (str): The collection name.

        Returns:
            VectorStoreIndex: The VectorStoreIndex from the QdrantVectorStore.
        """
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )

    def get_query_engine(
        self, type: Literal["origin", "contextual", "both"]
    ) -> BaseQueryEngine | dict[str, BaseQueryEngine]:
        """
        Get the query engine for the RAG.

        Args:
            type (Literal["origin", "contextual", "both"]): The type of RAG.

        Returns:
            BaseQueryEngine | dict[str, BaseQueryEngine]: The query engine.
        """

        if type == RAGType.ORIGIN:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.original_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.CONTEXTUAL:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.contextual_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.BOTH:
            return {
                "origin": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.original_rag_collection_name,
                ).as_query_engine(),
                "contextual": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.contextual_rag_collection_name,
                ).as_query_engine(),
            }

    def run_ingest(
        self,
        folder_dir: str | Path,
        type: Literal["origin", "contextual", "both"] = "contextual",
        docs_llama_pdf: list[Document] = [],
    ) -> None:
        """
        Run the ingest process for the RAG.

        Args:
            folder_dir (str | Path): The folder directory containing the papers.
            type (Literal["origin", "contextual", "both"]): The type to ingest. Default to `contextual`.
            docs_llama_pdf (list[Document]): List of documents from LlamaParse.
        """
        logger.info(f"Reading papers from {folder_dir}")

        if docs_llama_pdf != []:
            raw_documents, docs_llama_pdf = llama_parse_read_paper(folder_dir, type="thuann", exists=True)
        else:
            raw_documents, _ = llama_parse_read_paper(folder_dir, type="thuann", exists=False)
            raw_documents.extend(docs_llama_pdf)

        logger.info(f"Done reading papers from {folder_dir}")

        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        ingest_documents_raw = self.convert_format_ingest_data(raw_documents)

        if type == RAGType.ORIGIN:
            self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            print(len(contextual_documents), len(contextual_documents_metadata))

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.ingest_data(contextual_documents, type=RAGType.CONTEXTUAL)

            # Splitted data
            self.es.index_documents(
                contextual_documents_metadata, type=RAGType.CONTEXTUAL
            )

        logger.info(f"Done ingesting data to {type} RAG")
        logger.info(
            f"Done ingesting data to ElasticSearch {len(ingest_documents_raw)} raw, {len(raw_documents)} raw docs, {len(list(Path(folder_dir).iterdir()))} files"
        )
        
        return docs_llama_pdf

    def run_ingest_not_contextual(
        self,
        folder_dir: str | Path,
    ) -> None:
        """
        Run the ingest process for the RAG.

        Args:
            folder_dir (str | Path): The folder directory containing the papers.
            type (Literal["origin", "contextual", "both"]): The type to ingest. Default to `contextual`.
        """
        logger.info(f"Reading papers from {folder_dir}")

        raw_documents = llama_parse_read_paper(folder_dir, type="thuann")

        logger.info(f"Done reading papers from {folder_dir}")

        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []
        for each_splited in splited_documents:
            ingest_documents.extend(each_splited)
            for node in each_splited:
                doc_id = str(uuid.uuid4())
                documents_metadata.append(
                    DocumentMetadata(
                        doc_id=doc_id,
                        file_name=node.metadata["file_name"],
                        original_content=node.text,
                        contextualized_content="",
                    )
                )

                ingest_documents.append(
                    Document(
                    text=node.text,
                    metadata={
                        "doc_id": doc_id,
                        "file_name": node.metadata["file_name"],
                    },
                ),
                )

        # ingest_documents_raw = self.convert_format_ingest_data(raw_documents)

        self.ingest_data(ingest_documents, type=RAGType.ORIGIN)
        self.es.index_documents(documents_metadata, type=RAGType.ORIGIN)

        logger.info(f"Done ingesting data no contextual to {type} RAG")
        logger.info(
            f"Done ingesting data to ElasticSearch {len(raw_documents)} raw docs, {len(list(Path(folder_dir).iterdir()))} files"
        )

    def run_add_files(
        self,
        files_or_folders: list[str] | str,
        type: Literal["origin", "contextual", "both"],
    ):
        """
        Add files to the database.

        Args:
            files_or_folders (list[str]): List of file paths or paper folder to be ingested.
            type (Literal["origin", "contextual", "both"]): Type of RAG type to ingest.
        """
        # raw_documents = llama_parse_multiple_file(files_or_folders)
        # raw_documents = llama_parse_read_paper(
        #     files_or_folders, type="thuann"
        # )  # thuann when read pdf
        raw_documents = llama_read_txt_file(files_or_folders)
        # raw_documents = llama_parse_read_paper(files_or_folders, type="thuann")
        
        splited_documents = self.split_document(raw_documents)

        # ingest_documents_raw = self.convert_format_ingest_data(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        if type == RAGType.ORIGIN:
            self.insert_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.insert_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.insert_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.es.index_documents(
                contextual_documents_metadata, type=RAGType.CONTEXTUAL
            )

            # self.es.index_documents(ingest_documents_raw, type=RAGType.ORIGIN)
            
        logger.info(f"Done adding files to {type} RAG")

    def origin_rag_search(self, query: str) -> str:
        """
        Search the query in the Origin RAG.

        Args:
            query (str): The query to search.

        Returns:
            str: The search results.
        """

        index = self.get_query_engine(RAGType.ORIGIN)
        return index.query(query)

    def preprocess_abbreviations(self, query: str) -> str:
        """
        Preprocess the query by expanding the abbreviations.

        Args:
            query (str): The query to preprocess.

        Returns:
            str: The preprocessed query.
        """

        ABBREVIATION_DICT = get_abrreviate()

        for full_form, abbreviations in ABBREVIATION_DICT.items():
            for abbr in abbreviations:
                query = query.replace(abbr, full_form)

        return query

    def preprocess_query(self, query: str) -> str:
        """
        Reprompt the query by expanding the abbreviations and content by llm.

        Args:
            query (str): The query to preprocess.

        Returns:
            str: The preprocessed query.
        """
        # current_time = datetime.now()
        # year = current_time.year - 1
        # current_time_str = str(year)
        # current_time_str = "2024"

        # print(f"Current year: {current_time_str}")

        ABBREVIATION_LIST = str(get_abrreviate())

        prompt = ChatPromptTemplate(
            message_templates=[
                # ChatMessage(
                #     role="system",
                #     content=(
                #         "You are a helpful assistant for supportive Information Technology university admissions counselor chatbot, specializing in understanding abbreviations.\n"
                #         # "When encountering an abbreviation, expand it to its full meaning based on the context.\n"
                #         "If the user does not specify a different year, default to the current year: {current_time}"
                #         "Identify and expand any abbreviations in the query, especially those must be in Vietnamese.\n"
                #         "You only need to respond to the user's expanded query."
                #     ),
                # ),
                ChatMessage(
                    role="system",
                    content=(
                        "You are a helpful assistant for a understand query from the user when ask in Information Technology University (UIT) admissions counselor chatbot. Your primary task is to understand and expand abbreviations within the user's query and ensure clarity in its meaning.\n"
                        # "If the user does not specify a different year, default to the current year: {current_time}.\n"
                        "Your response should only be the expanded and clarified version of the user's query, formatted in proper Vietnamese for effective document retrieval."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=EXPAND_QUERY_PROMPT.format(
                        abbreviations_str=ABBREVIATION_LIST,
                        context_str=query,
                        # current_time_str=current_time_str,
                    ),
                ),
            ]
        )

        messages = prompt.format_messages()

        expanded_query = self.llm.chat(messages).message.content

        print(f"Original query: {query}")
        print(f"Expanded query: {expanded_query}")

        return expanded_query

    async def apreprocess_query(self, query: str) -> str:
        """
        Reprompt the query by expanding the abbreviations and content by llm.

        Args:
            query (str): The query to preprocess.

        Returns:
            str: The preprocessed query.
        """
        ABBREVIATION_LIST = str(get_abrreviate())

        prompt = ChatPromptTemplate(
            message_templates=[
                # ChatMessage(
                #     role="system",
                #     content=(
                #         "You are a helpful assistant for supportive Information Technology university admissions counselor chatbot, specializing in understanding abbreviations.\n"
                #         # "When encountering an abbreviation, expand it to its full meaning based on the context.\n"
                #         "If the user does not specify a different year, default to the current year: {current_time}"
                #         "Identify and expand any abbreviations in the query, especially those must be in Vietnamese.\n"
                #         "You only need to respond to the user's expanded query."
                #     ),
                # ),
                ChatMessage(
                    role="system",
                    content=(
                        "You are a helpful assistant for a supportive Information Technology university admissions counselor chatbot. Your primary task is to understand and expand abbreviations within the user's query and ensure clarity in its meaning.\n"
                        # "If the user does not specify a different year, default to the current year: {current_time}.\n"
                        "Your response should only be the expanded and clarified version of the user's query, formatted in proper Vietnamese for effective document retrieval."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=EXPAND_QUERY_PROMPT.format(
                        abbreviations_str=ABBREVIATION_LIST,
                        context_str=query,
                    ),
                ),
            ]
        )

        messages = prompt.format_messages()

        expanded_query = (await self.llm.achat(messages)).message.content

        print(f"Original query: {query}")
        print(f"Expanded query: {expanded_query}")

        return expanded_query

    def contextual_rag_search(
        self, query: str, k: int = 10, debug: bool = False, query_expand: bool = True
    ) -> str:
        """
        Search the query with the Contextual RAG.

        Args:
            query (str): The query to search.
            k (int): The number of documents to return. Default to `150`.
            debug (bool): Debug mode

        Returns:
            str: The search results.
        """
        start_time_all = time.time()

        start_time = time.time()
        
        if query_expand:
            query = self.preprocess_query(query)
        
        if "điểm tuyển sinh" in query:
            query = query.replace("điểm tuyển sinh", "điểm chuẩn")
            if "ngành" not in query:
                query += " các ngành"

        index_contextual = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name
        )

        # index_original = self.get_qdrant_vector_store_index(
        #     self.qdrant_client, self.setting.original_rag_collection_name
        # )

        index_original = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name_2
        )

        retriever_contextual = VectorIndexRetriever(
            index=index_contextual,
            similarity_top_k=k,
        )

        retriever_original = VectorIndexRetriever(
            index=index_original,
            similarity_top_k=k,
        )

        query_engine_contextual = RetrieverQueryEngine(retriever=retriever_contextual)
        query_engine_original = RetrieverQueryEngine(retriever=retriever_original)

        logger.info(
            f"Querying ..., loading data from database {time.time() - start_time}"
        )

        start_time = time.time()
        semantic_results_contextual: Response = query_engine_contextual.query(query)
        semantic_results_original: Response = query_engine_original.query(query)
        

        dict_filename = {}
        nodes_contextual = semantic_results_contextual.source_nodes
        nodes_original = semantic_results_original.source_nodes
        
        combined_nodes_contextual = []
        for node in nodes_contextual:
            combined_nodes_contextual.append(
                NodeWithScore(
                    node=TextNode(
                        text=node.get_content(),
                        metadata={
                            "doc_id": "",
                            "file_name": node.metadata["file_name"],
                            "search": "both",
                        },
                    ),
                    score=node.score,
                )
            )
            
            
        combined_nodes_original = []
        for node in nodes_original:
            combined_nodes_original.append(
                NodeWithScore(
                    node=TextNode(
                        text=node.get_content(),
                        metadata={
                            "doc_id": "",
                            "file_name": node.metadata["file_name"],
                            "search": "both",
                        },
                    ),
                    score=node.score,
                )
            )

        logger.info(f"BM25 Querying ..., search semantic {time.time() - start_time}")
        start_time = time.time()
        bm25_results = self.es.search(query, k=k)

        bm25_doc_id = []
        for result in bm25_results:
            dict_filename[result.doc_id] = result.file_name
            bm25_doc_id.append(result.doc_id)

        logger.info(f"Done BM25 Querying ..., search bm25 {time.time() - start_time}")
        start_time = time.time()

        combined_nodes: list[NodeWithScore] = []

        retrieved_nodes_original = sorted(combined_nodes_original, key=lambda x: x.score, reverse=True)
        retrieved_nodes_contextual = sorted(combined_nodes_contextual, key=lambda x: x.score, reverse=True)

        def find_absolute_path(title: str) -> str:
            """get data from the title file"""
            path_collection_data = Path(self.setting.collection_data)
            for file in path_collection_data.iterdir():
                if title in file.name:
                    return str(file)
                if file.suffix == ".txt":
                    with open(file, "r") as f:
                        links_html = f.readlines()
                        for link in links_html:
                            if title in link:
                                return str(link)
            return ""

        method_search, judge_mrr_original = [], []
        context_str_original = ""
        for n in retrieved_nodes_original:
            document_retrieved = n.node.text
            document_retrieved = document_retrieved.replace("\n", " ")
            document_retrieved = document_retrieved.replace("\t", " ")
            document_retrieved = document_retrieved.replace("\xa0", " ")

            method_search.append(n.node.metadata["search"])
            
            judge_mrr_original.append({
                "context": document_retrieved,
                "file_name": n.node.metadata["file_name"],
                "score": n.score,
            })
            
            
            
        method_search, judge_mrr_contextual = [], [] 
        context_str_contextual = ""
            
        for n in retrieved_nodes_contextual:
            document_retrieved = n.node.text
            document_retrieved = document_retrieved.replace("\n", " ")
            document_retrieved = document_retrieved.replace("\t", " ")
            document_retrieved = document_retrieved.replace("\xa0", " ")

            method_search.append(n.node.metadata["search"])
            
            judge_mrr_contextual.append({
                "context": document_retrieved,
                "file_name": n.node.metadata["file_name"],
                "score": n.score,
            })

        ABBREVIATION_LIST = str(get_abrreviate())

        prompt_original = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(SYSTEM_QA_PROMPT),
                ),
                ChatMessage(
                    role="user",
                    content=QA_PROMPT.format(
                        abbreviations_str=ABBREVIATION_LIST,
                        format_answer="""{"answer": Your answer here. You can get more informations in resources; "resources": [Link to the resources]}""",
                        context_str=context_str_original,
                        query_str=query,
                    )
                    + "\nVui lòng liên hệ phòng tư vấn tuyển sinh UIT để biết thêm chi tiết. \n\nPlease respond in JSON format.",
                ),
            ]
        )

        prompt_contextual = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(SYSTEM_QA_PROMPT),
                ),
                ChatMessage(
                    role="user",
                    content=QA_PROMPT.format(
                        abbreviations_str=ABBREVIATION_LIST,
                        format_answer="""{"answer": Your answer here. You can get more informations in resources; "resources": [Link to the resources]}""",
                        context_str=context_str_contextual,
                        query_str=query,
                    )
                    + "\nVui lòng liên hệ phòng tư vấn tuyển sinh UIT để biết thêm chi tiết. \n\nPlease respond in JSON format.",
                ),
            ]
        )

        messages_original = prompt_original.format_messages(chunk_and_resource=CHUNK_AND_RESOURCE)
        messages_contextual = prompt_contextual.format_messages(chunk_and_resource=CHUNK_AND_RESOURCE)

        response_original = self.llm.chat(
            messages_original, response_format={"type": "json_object"}
        ).message.content
        
        response_contextual = self.llm.chat(
            messages_contextual, response_format={"type": "json_object"}
        ).message.content

        logger.info(f"Done all {time.time() - start_time_all}")
        print(f"Contexts: {response_original}")
        print(f"Contexts: {response_contextual}")

        return response_contextual, response_original, judge_mrr_contextual, judge_mrr_original

    async def acontextual_compress(
        self,
        query: str,
        retrieved_nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """
        Compress the retrieved nodes.

        Args:
            query (str): The query to compress.
            retrieved_nodes (list[NodeWithScore]): The retrieved nodes to compress.

        Returns:
            list[NodeWithScore]: The compressed nodes.
        """
        messages = [
            [
                ChatMessage(
                    role="system",
                    content=contextual_compression_system_prompt_template,
                ),
                ChatMessage(
                    role="user",
                    content=contextual_compression_prompt_template.format(
                        question=query,
                        context=node.text,
                    ),
                ),
            ]
            for node in retrieved_nodes
        ]

        requests = [
            self.contextual_compressor_llm.achat(message) for message in messages
        ]

        logger.info(f"Compress {len(retrieved_nodes)} nodes")
        start_time = time.time()

        responses = await asyncio.gather(*requests)

        logger.info(f"Total: {time.time() - start_time}")

        compressed_nodes = []

        for response, node in zip(responses, retrieved_nodes):
            compressed_nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=response.message.content,
                        metadata=node.node.metadata,
                    ),
                    score=node.score,
                )
            )

        return compressed_nodes

    async def acontextual_rag_search(
        self, query: str, k: int = 2, debug: bool = False, use_compressor: bool = False
    ) -> str:
        """
        Search the query with the Contextual RAG.

        Args:
            query (str): The query to search.
            k (int): The number of documents to return. Default to `150`.
            debug (bool): Debug mode

        Returns:
            str: The search results.
        """
        start_time_all = time.time()

        # query = await self.apreprocess_query(query)

        start_time = time.time()

        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight

        index = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=k,
        )

        query_engine = RetrieverQueryEngine(retriever=retriever)

        logger.info(
            f"Querying ..., loading data from database {time.time() - start_time}"
        )

        start_time = time.time()
        semantic_results: Response = query_engine.query(query)

        dict_filename = {}
        semantic_doc_id = []
        nodes = semantic_results.source_nodes
        for node in nodes:
            dict_filename[node.metadata["doc_id"]] = node.metadata["file_name"]
            semantic_doc_id.append(node.metadata["doc_id"])

        logger.info(f"BM25 Querying ..., search semantic {time.time() - start_time}")
        start_time = time.time()
        bm25_results = self.es.search(query, k=k)

        bm25_doc_id = []
        for result in bm25_results:
            dict_filename[result.doc_id] = result.file_name
            bm25_doc_id.append(result.doc_id)

        logger.info(f"Done BM25 Querying ..., search bm25 {time.time() - start_time}")
        start_time = time.time()

        combined_nodes: list[NodeWithScore] = []

        combined_ids = list(set(semantic_doc_id + bm25_doc_id))

        def get_content_by_doc_id(doc_id: str):
            for node in nodes:
                if node.metadata["doc_id"] == doc_id:
                    return node.text
            return ""

        # Compute score according to: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
        bm25_count = 0
        semantic_count = 0
        both_count = 0
        results_content = []
        for id in combined_ids:
            score = 0
            content = ""

            if id in semantic_doc_id:
                index = semantic_doc_id.index(id)
                score += semantic_weight * (1 / (index + 1))

                semantic_count += 1

                content = get_content_by_doc_id(id)

            if id in bm25_doc_id:
                index = bm25_doc_id.index(id)
                score += bm25_weight * (1 / (index + 1))
                bm25_count += 1

                if content == "":
                    content = (
                        bm25_results[index].contextualized_content
                        + "\n\n"
                        + bm25_results[index].content
                    )

            if id in semantic_doc_id and id in bm25_doc_id:
                both_count += 1

            combined_nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=content,
                        metadata={
                            "doc_id": id,
                            "search": (
                                "both"
                                if id in semantic_doc_id and id in bm25_doc_id
                                else "elastic" if id in bm25_doc_id else "semantic"
                            ),
                        },
                    ),
                    score=score,
                )
            )

            results_content.append(content)

        if debug:
            logger.info(
                "Semantic count: %s, BM25 count: %s, Both count: %s",
                semantic_count,
                bm25_count,
                both_count,
            )

        # query_bundle = QueryBundle(query_str=query)

        logger.info(f"Done combining nodes {time.time() - start_time}")

        retrieved_nodes = sorted(
            combined_nodes, key=lambda x: float(x.score), reverse=True
        )
        # retrieved_nodes = self.reranker.postprocess_nodes(combined_nodes, query_bundle)

        if use_compressor:
            print(f"Retrieved nodes: {retrieved_nodes}")
            retrieved_nodes = await self.acontextual_compress(query, retrieved_nodes)

        method_search = []
        context_str = ""
        for n in retrieved_nodes:
            score = float(n.score)

            title = dict_filename[n.node.metadata["doc_id"]]

            title = title.split("\n")[0]

            title = get_title_private(title)

            document_retrieved = n.node.text
            document_retrieved = document_retrieved.replace("\n", " ")
            document_retrieved = document_retrieved.replace("\t", " ")
            document_retrieved = document_retrieved.replace("\xa0", " ")

            if not title.startswith("https"):
                title = "files/" + title
            context_str += f"Chunk: {document_retrieved} \n Resource: {title} \n ======================== \n\n"
            method_search.append(n.node.metadata["search"])

        ABBREVIATION_LIST = str(get_abrreviate())

        prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(SYSTEM_QA_PROMPT),
                ),
                # ChatMessage(
                #     role="user",
                #     content=QA_PROMPT.format(
                #         abbreviations_str=ABBREVIATION_LIST,
                #         format_answer="""answer - Your answer here. You can get more informations in resources; resources- [Link to the resources]""",
                #         context_str=context_str,
                #         query_str=query,
                #     ) + "\nVui lòng liên hệ phòng tư vấn tuyển sinh UIT để biết thêm chi tiết. \n\nPlease respond in JSON format.",
                # ),
                ChatMessage(
                    role="user",
                    content=QA_PROMPT.format(
                        abbreviations_str=ABBREVIATION_LIST,
                        format_answer="""{"answer": Your answer here. You can get more informations in resources; "resources": [Link to the resources]}""",
                        context_str=context_str,
                        query_str=query,
                    )
                    + "\nVui lòng liên hệ phòng tư vấn tuyển sinh UIT để biết thêm chi tiết. \n\nPlease respond in JSON format.",
                ),
            ]
        )

        messages = prompt.format_messages(chunk_and_resource=CHUNK_AND_RESOURCE)

        response = (
            await self.llm.achat(messages, response_format={"type": "json_object"})
        ).message.content

        logger.info(f"Done all {time.time() - start_time_all}")
        print(f"Contexts: {context_str}")
        return response
