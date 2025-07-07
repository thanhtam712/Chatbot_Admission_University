from pathlib import Path

import qdrant_client
from qdrant_client.http import models
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    text,
    insert,
)
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI

# Connect to Qdrant
client_base = QdrantClient("192.168.28.151", port=8888)
client = qdrant_client.QdrantClient(url="192.168.28.151", port=8888)
vector_store = QdrantVectorStore(collection_name="contextual_2025", client=client)

total_folder = ["./raw_data_2024", "./raw_data_2025", "./add_data/diemchuan"]
total_cite = ["./data_filter_2024", "./add_data/2025"]
list_files = []
for folder in total_folder:
    list_files += [file.stem for file in Path(folder).iterdir()]

print(f"Total files: {len(list_files)}")
    
dict_cite = {}
for folder in total_cite:
    for file in Path(folder).iterdir():
        if file.suffix not in [".txt"]:
            # if file.stem in list_files:
            dict_cite[file.name] = file.stem
            # else:
                # print(f"File {file.name} not in list_files")
                
        elif file.suffix == ".txt":
            data_urls = open(file, "r", encoding="utf-8").readlines()
            for data_url in data_urls:
                # if data_url.strip().split("/")[-1] in list_files:
                dict_cite[data_url.strip()] = data_url.strip().split("/")[-1]
                # else:
                    # print(f"Url {data_url.strip()} not in list_files")
                    
print(f"Total files to process: {len(dict_cite)}")

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

table_name = "context_table_3"
context_table = Table(
    table_name,
    metadata_obj,
    Column("file_name", String(16)),
    Column("id", String(16), primary_key=True),
    Column("context", String(2048), nullable=False),
)
metadata_obj.create_all(engine)

sql_database = SQLDatabase(engine, include_tables=["context_table_3"])

for key, value in dict_cite.items():
    file_name = key
    response = client_base.scroll(
        collection_name="contextual_2025",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file_name),
                ),
            ]
        ),
        limit=20,
        with_payload=True,
    )
    
    for point in response[0]:
        node_id = point.id
        doc_id = point.payload["doc_id"]
        content = vector_store.get_nodes(node_ids=[node_id], filters=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file_name),
                ),
            ]
        ))
        print(f"Node ID: {node_id}")
        for c in content:
            print(c[0].start_char_idx)
            input()
        
        row = {
            "file_name": file_name,
            "id": node_id,
            "context": content[0].text,
        }
        
        # stmt = insert(context_table).values(**row)
        # with engine.begin() as connection:
        #     cursor = connection.execute(stmt)
        
        print(f"Done inserting {file_name} with id {node_id}")
            
def retrieve_context(file_name: str):
    """
    Retrieve context from the context table.
    Args:
        file_name (str): The name of the file to retrieve context for.
    
    Returns:
        list: A list of context strings.
    """
    stmt = select(
        context_table.c.file_name,
        # context_table.c.id,
        context_table.c.context,
    ).select_from(context_table)
    
    stmt = select(
        context_table.c.context,
        # context_table.c.id,
        context_table.c.file_name,
    ).where(context_table.c.file_name == file_name)
    
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        print([row.context for row in results])
        input("Press Enter to continue...")
    # with engine.connect() as con:
    #     rows = con.execute(text("SELECT context from context_table"))
    #     for row in rows:
    #         print(row)

    # with engine.connect() as connection:
    #     results = connection.execute(stmt).fetchall()
    #     print(results)

# retrieve_context("https://tuyensinh.uit.edu.vn/2025-du-kien-phuong-thuc-tuyen-sinh-nam-2025")
