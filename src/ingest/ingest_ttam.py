import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding import RAG
from src.settings import Settings

load_dotenv(dotenv_path="./uit_chatbot/.env.production")

def load_parser():
    parser = argparse.ArgumentParser(description="Ingest data")
    parser.add_argument(
        "--folder-dir",
        type=str,
        help="Path to the folder containing the documents or path to the file containing links",
    )
    parser.add_argument(
        "--type",
        choices=["origin", "contextual", "both"],
        required=True,
    )
    return parser.parse_args()


def main():
    args = load_parser()

    setting = Settings()
    docs_llama_pdf = []

    for folder in tqdm(Path(args.folder_dir).iterdir()):
        setting.original_rag_collection_name = "multi_" + folder.stem
        setting.contextual_rag_collection_name = "multi_" + folder.stem
        setting.elastic_search_index_name = "multi_" + folder.stem
        
        rag = RAG(setting=setting)
        # rag.run_add_files(files_or_folders=args.folder_dir, type=args.type)
        # rag.run_add_files(files_or_folders=list(Path(args.folder_dir).iterdir()), type=args.type)

        # rag.run_ingest_not_contextual(folder_dir=args.folder_dir)
        
        docs_llama_pdf = rag.run_ingest(folder_dir=folder, type=args.type, docs_llama_pdf=docs_llama_pdf)
            
if __name__ == "__main__":
    main()
