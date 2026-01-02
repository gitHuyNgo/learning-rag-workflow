import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List

import weaviate
from weaviate.classes.query import MetadataQuery

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAIResponses
from llama_index.core.llms import ChatMessage

from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi




DATA_PATH = Path("./test_data")
CLASS_NAME = "document_chunk_embedding"
RERANK_MODEL = "BAAI/bge-reranker-large"
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")




def init_weaviate():
    client = weaviate.connect_to_local()
    return client
        
def vector_retrieve(client, query: str, top_k: int = 10):
    embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    query_vector = embed_model.get_text_embedding(query)
    collection = client.collections.use(CLASS_NAME)

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    return [obj.properties for obj in response.objects]


def bm25_rerank(query: str, docs: List[dict], top_k: int = 5):
    corpus = [d["text"] for d in docs]
    tokenized = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, _ in ranked[:top_k]]
    

def bge_rerank(query: str, docs: List[dict], top_k: int = 3):
    reranker = FlagReranker(RERANK_MODEL, use_fp16=True)

    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.compute_score(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, _ in ranked[:top_k]]


def transform(docs: List[dict]) -> str:
    context = []
    for i, d in enumerate(docs, 1):
        part = f"""
            [Chunk {i}]
            Source: {d['source']}
            Content:
            {d['text']}
        """
        context.append(part.strip())
    return "\n\n".join(context)


def answer(client, query: str, llm):
    vec_docs = vector_retrieve(client, query)
    bm25_docs = bm25_rerank(query, vec_docs)
    final_docs = bge_rerank(query, bm25_docs)
    docs = transform(final_docs)

    messages = [
        ChatMessage(
            role="system",
            content="You are an assistant that helps to answer questions based on the provided context",
        ),
        ChatMessage(
            role="user",
            content=f"Answer the following question using the provided context.\n\nQuestion: {query}\n\nContext:\n{docs}",
        ),
    ]
    resp = llm.chat(messages)
    print(resp)


def main():
    client = init_weaviate()
    llm = OpenAIResponses(
        model="gpt-5.1",
        api_key=OPENAI_API_KEY
    )
    try:
        while True:
            q = input("\nAsk (or exit): ")
            if q.lower() == "exit":
                break
            answer(client, q, llm)
    finally:
        client.close()


if __name__ == "__main__":
    main()