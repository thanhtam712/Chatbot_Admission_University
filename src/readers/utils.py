import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.constants import SUPPORTED_FILE_EXTENSIONS, SUPPORTED_LINK_EXTENSIONS
from src.utils import get_formatted_logger

logger = get_formatted_logger(__file__)


def check_valid_extenstion(file_path: str | Path) -> bool:
    """
    Check if the file extension is supported

    Args:
        file_path (str | Path): File path to check

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    return Path(file_path).suffix in SUPPORTED_FILE_EXTENSIONS


def check_valid_link(link: str) -> bool:
    """
    Check if the link is supported

    Args:
        link (str): Link to check

    Returns:
        bool: True if the link is supported, False otherwise.
    """
    return link.split(":")[0] in SUPPORTED_LINK_EXTENSIONS


def get_files_from_folder_or_file_paths(files_or_folders: list[str]) -> list[str]:
    """
    Get all files from the list of file paths or folders

    Args:
        files_or_folders (list[str]): List of file paths or folders

    Returns:
        list[str]: List of valid file paths.
    """
    files = []

    for file_or_folder in files_or_folders:
        if Path(file_or_folder).is_dir():
            files.extend(
                [
                    str(file_path.resolve())
                    for file_path in Path(file_or_folder).rglob("*")
                    if check_valid_extenstion(file_path)
                ]
            )

        else:
            if check_valid_extenstion(file_or_folder):
                files.append(str(Path(file_or_folder).resolve()))
            else:
                logger.warning(f"Unsupported file extension: {file_or_folder}")

    return files


def get_files_from_list_links(links: list[str]) -> list[str]:
    """
    Get all links from the list of links

    Args:
        links (list[str]): List of links

    Returns:
        list[str]: List of valid file paths.
    """
    links_return = []

    for link in links:
        if check_valid_link(link):
            links_return.append(link.strip())
        else:
            logger.warning(f"Unsupported link extension: {link}")

    return links_return

def classify_files(folder_path: str | Path) -> dict[str, list[str]]:
    """
    Classify files in a folder to pdfs, docxs, htmls, and images

    Args:
        folder_path (str | Path): Path to the folder
    """
    list_pdfs, list_docxs, list_htmls, list_imgs, list_llama_pdf = [], [], [], [], []
    dict_classify = {
        "pdf": list_pdfs,
        "docx": list_docxs,
        "html": list_htmls,
        "img": list_imgs,
        "llama_pdf": list_llama_pdf}
    
    for file in Path(folder_path).iterdir():
        if file.suffix in [".png", ".jpg", ".jpeg"]:
            dict_classify["img"].append(str(file.resolve()))
            
        elif file.suffix == ".txt":
            with open(file, "r") as f:
                link_webs = f.readlines()
                dict_classify["html"].extend(get_files_from_list_links(link_webs))
        
        # .docx and .pdf
        elif file.suffix in SUPPORTED_FILE_EXTENSIONS:
            if file.name == "CAM_NANG_UIT.pdf":
                dict_classify["llama_pdf"].append(str(file.resolve()))
            else:
                dict_classify[file.suffix[1:]].append(str(file.resolve()))
    return dict_classify
