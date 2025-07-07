from pathlib import Path
from typing import Optional, List, Dict 

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import PDFReader


class PaperReader(BaseReader):
    """PDF parser and Image parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Parse file pdf."""
        
        if file.suffix == ".pdf":
            documents = PDFReader().load_data(file)
            for doc in documents:
                doc.metadata["file_name"] = file.name
                
                print(f"Document {doc}")

        return documents
