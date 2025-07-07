from pathlib import Path
from markitdown import MarkItDown
from typing import Optional, List, Dict

from src.settings import Settings

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import TokenTextSplitter


class MarkItDownReader(BaseReader):
    """Markdown parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Parse file."""
        settings = Settings()

        document = MarkItDown().convert(str(file)).text_content

        docs_splitted = TokenTextSplitter(
            chunk_size=settings.num_token_split_docx, separator=" "
        ).get_nodes_from_documents([Document(text=document)])

        documents: List[Document] = []
        for idx, doc in enumerate(docs_splitted):
            documents.append(
                Document(
                    text=doc.text,
                    metadata={"file_name": Path(file).name, "page_number": idx + 1},
                )
            )
            
        # documents = Document(text=document, metadata={"file_name": Path(file).name})

        return [documents]
