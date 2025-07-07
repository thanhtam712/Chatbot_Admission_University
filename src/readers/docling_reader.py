from typing import Optional, List, Dict
from docling.document_converter import DocumentConverter

from src.settings import Settings

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.text_splitter import TokenTextSplitter

settings = Settings()

class DoclingReader(BaseReader):
    """Docling parser."""

    def load_data(
        self,
        link: str | List[str],
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Parse link html to get the text.
        
        With html link, it will split the html into multiple documents.
        
        Args:
            link (str | List[str]): List of link html or single link html.
            extra_info (Optional[Dict], optional): Extra information. Defaults to None.
            
        Returns:
            List[Document]: List of documents.
        """
        document = DocumentConverter().convert_all(link)
        documents: List[Document] = []

        for idxs, doc in enumerate(document):
            doc = doc.document.export_to_markdown()

            docs_splitted = TokenTextSplitter(
                chunk_size=settings.num_token_split_docx, separator=" "
            ).get_nodes_from_documents([Document(text=doc)])

            for idx, doc in enumerate(docs_splitted):
                documents.append(
                    Document(
                        text=doc.text,
                        metadata={
                            "file_name": link[idxs],
                            "page_number": idx + 1,
                        },
                    )
                )
            
            # documents.append(Document(text=doc, metadata={"file_name": link[idxs]}))

        return documents
