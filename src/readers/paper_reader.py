import polars as pl 
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv
from natsort import natsorted
from typing import Literal

from llama_parse import LlamaParse
import google.generativeai as genai
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

from .utils import get_files_from_folder_or_file_paths, get_formatted_logger, classify_files
from .thuann_reader import ThuaNNPdfReader
from .markitdown_reader import MarkItDownReader
from .docling_reader import DoclingReader

load_dotenv()
logger = get_formatted_logger(__name__)


def gemini_read_paper_content(
    paper_dir: Path | str, save_dir: Path | str = "output"
) -> list[Document]:
    """
    Read the content of the paper using the Gemini.

    Args:
        paper_dir (str | Path): Path to the directory containing the papers
        save_dir (str | Path): Path to the directory to save the extracted content
    Returns:
        list[Document]: List of documents from all papers.
    """
    paper_dir = Path(paper_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    paper_file = [str(file) for file in paper_dir.glob("*.pdf")]
    model = genai.GenerativeModel("gemini-1.5-flash")

    documents: list[Document] = []

    for file in tqdm(paper_file):
        assert isinstance(file, str)

        file_name = Path(file).stem + ".txt"

        ic(file)

        pdf_file = genai.upload_file(file)
        response = model.generate_content(
            [
                r"""Extract all content from this paper, must be in human readable order. Each paper content is put in seperate <page></page> tag""",
                pdf_file,
            ]
        )
        documents.append(Document(text=response.text))

        with open(save_dir / file_name, "w") as f:
            f.write(response.text)

    return documents


def gemini_read_paper_content_single_file(file_path: Path | str) -> Document:
    """
    Read the content of one paper using the Gemini.

    Args:
        file_path (str | Path): Path to the paper file
    Returns:
        Document: Document object from the paper.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    pdf_file = genai.upload_file(file_path)
    response = model.generate_content(
        [
            r"""Extract all content from this paper, must be in human readable order. Each paper content is put in seperate <page></page> tag""",
            pdf_file,
        ]
    )

    return Document(text=response.text)


def llama_parse_read_paper(paper_dir: Path | str | list[str], type: Literal["llama", "thuann"] = "thuann", exists: bool = False) -> list[Document]:
    """
    Read the content of the paper using  LlamaParse.

    Args:
        paper_dir (str | Path | list[str]): Path to the directory all containing the papers.
        type (str): Type of the parser to use read file pdf.
    Returns:
        list[Document]: List of documents from all papers.
        exists (bool): True if the file special pdf read, False otherwise.
    """
    ic(paper_dir)

    documents: list[Document] = []

    dict_classify = classify_files(paper_dir)
    list_llama_pdf = dict_classify["llama_pdf"]
    list_pdfs = dict_classify["pdf"]
    list_docxs = dict_classify["docx"]
    list_htmls = dict_classify["html"]
    list_imgs = dict_classify["img"]
    
    print(f"List pdfs: {list_pdfs}")
    print(f"List llama_pdf: {list_llama_pdf}")
    
    docs_llama_pdf = []
    if len(list_pdfs) > 0 or len(list_docxs) > 0 or len(list_imgs) > 0 or len(list_llama_pdf) > 0:
        paper_dir = list_pdfs + list_docxs
        docs, docs_img, docs_llama_pdf = [], [], []
        
        parser_pdf_thuann = ThuaNNPdfReader()
        parser_pdf_llama = PDFReader()
        parser_md = MarkItDownReader()
        
        valid_files = get_files_from_folder_or_file_paths(paper_dir)

        ic(valid_files)

        if len(valid_files) > 0:
            if type == "llama":
                reader = SimpleDirectoryReader(
                    input_files=valid_files,
                    file_extractor={".pdf": parser_pdf_llama, ".docx": parser_md},
                )
            elif type == "thuann":
                reader = SimpleDirectoryReader(
                    input_files=valid_files,
                    file_extractor={".pdf": parser_pdf_thuann, ".docx": parser_md},
                )
            else:
                raise ValueError("Type of parser is not supported.")
            
            docs = reader.load_data(show_progress=True)

        # Handle for special case: pdfs that are not supported by LlamaParse 
        if len(list_llama_pdf) > 0 and not exists:
            reader_llama = SimpleDirectoryReader(
                input_files=list_llama_pdf,
                file_extractor={".pdf": parser_pdf_llama},
            )

            docs_llama_pdf = reader_llama.load_data(show_progress=True)
            
        if len(list_imgs) > 0:
            reader_img = SimpleDirectoryReader(
                input_files=list_imgs,
                file_extractor={".png": parser_pdf_thuann, ".jpg": parser_pdf_thuann, ".jpeg": parser_pdf_thuann},
            )

            docs_img = reader_img.load_data(show_progress=True)
            
        documents.extend(docs)
        documents.extend(docs_img)
        documents.extend(docs_llama_pdf)
        
        logger.info(f"Successfully parsed the document from the file: {valid_files}")
            
        if len(docs_img) != len(list_imgs):
            logger.warning(f"Cannot parse the document from the image: {list_imgs}")
        else:
            logger.info(f"Successfully parsed the document from the image: {list_imgs}")

    if len(list_htmls) > 0:

        ic(list_htmls)

        parser_web = DoclingReader()

        for link in tqdm(list_htmls, total=len(list_htmls)):
            doc = parser_web.load_data([link])
            documents.extend(doc)

            if len(doc) == 0:
                logger.warning(f"Cannot parse the document from the link: {link}")
            else:
                logger.info(f"Successfully parsed the document from the link: {link}")
    
    print(documents)
            
    return documents, docs_llama_pdf

def llama_read_csv_file(file_path: Path | str | list[str]) -> list[Document]:
    """
    Read the content of the papers from the CSV file using LlamaParse.
    
    Args:
        file_path (str | Path): Path to the CSV file.
    
    Returns:
        list[Document]: List of documents from all papers.
    """
    
    docs = []

    for file_csv in file_path:
        title = Path(file_csv).stem.replace("_", " ")
        pthuc = Path(file_csv).stem.split("_")[-1]
        year = Path(file_csv).stem.split("_")[0]
        doc = f"Điểm chuẩn tuyển sinh các ngành năm {title} theo phương thức {pthuc}\n"
        doc += f"Điểm tuyển sinh theo phương thức {pthuc} năm {year}\n"
        data = pl.read_csv(file_csv, truncate_ragged_lines=True)
        headers = data.columns[0].split("\t")
        
        for row in data.iter_rows(named=False): 
            row = row[0].split("\t")
            for item, col in zip(row, headers):
                doc += f"{col}: {item}\t"
            doc += "\n"
        print(f"Doc in file {Path(file_csv).name}: {doc}")
        
        docs.append(Document(text=doc, metadata={"file_name": Path(file_csv).name}))
    
    return docs

def llama_read_txt_file(file_path: Path | str | list[str]) -> list[Document]:
    """
    Read the content of the papers from the txt file using LlamaParse.
    
    Args:
        file_path (str | Path): Path to the TXT file.
    
    Returns:
        list[Document]: List of documents from all papers.
    """
    
    if isinstance(file_path, (str, Path)):
        file_path = [file_path]

    docs = []
    for file_txt in file_path:
        ic(file_txt)
        with open(file_txt, "r") as f:
            data = f.readlines()
            doc = ""
            for line in data:
                if line.strip() == "-------------------------------------------------------------":
                    docs.append(Document(text=doc, metadata={"file_name": Path(file_txt).name}))
                    print(doc)
                    print("Done read doc")
                    doc = ""
                else:
                    doc += line

    return docs
    
def llama_parse_single_file(file_path: Path | str) -> Document:
    """
    Read the content of one paper using LlamaParse.

    Args:
        file_path (str | Path): Path to the paper file.
    Returns:
        Document: Document object from the paper.
    """
    parser = LlamaParse(result_type="markdown")

    file_path = Path(file_path)

    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return documents


def llama_parse_multiple_file(files_or_folder: list[str]) -> list[Document]:
    """
    Read the content of multiple papers using LlamaParse.

    Args:
        files_or_folder (list[str]): List of file paths or folder paths containing the papers.
    Returns:
        list[Document]: List of documents from all papers.
    """
    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    ic(valid_files)

    parser = LlamaParse(result_type="markdown")
    # parser = PDFReader()

    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return documents
