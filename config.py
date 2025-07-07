# llm_config = dict(
#     service="groq",
#     model="llama3-70b-8192",
# )
llm_config = dict(
    service="openai",
    model="gpt-4.1-mini",
)
# llm_config = dict(
#     service="gemini",
#     model="models/gemini-1.5-flash",
# )


# embedding_config = dict(
#     service="huggingface",
#     model="BAAI/bge-m3",
# )
embedding_config = dict(
    service="openai",
    model="text-embedding-3-small",
)
# embedding_config = dict(
#     service="openai",
#     model="text-embedding-ada-002",
# )


# reranker_config = dict(service="cohere", model="rerank-multilingual-v3.0")
reranker_config = dict(service="rankgpt", model="gpt-4o-mini")
