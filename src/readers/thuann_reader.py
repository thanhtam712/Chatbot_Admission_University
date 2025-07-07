import sys
from pathlib import Path
from typing import Optional, List, Dict

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


from .nonpdf import convert, convert_imgs
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.constants import SUPPORTED_FILE_EXTENSIONS


class ThuaNNPdfReader(BaseReader):
    """PDF parser and Image parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        **kwargs,
    ) -> List[Document]:
        """Parse file pdf."""
        
        if file.suffix == ".pdf":
            documents = convert(str(file))
        elif file.suffix in SUPPORTED_FILE_EXTENSIONS:
            documents = convert_imgs([str(file)])

        return documents
