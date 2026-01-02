import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

import weaviate
from weaviate.classes.config import Property, DataType, Configure

from docling.document_converter import DocumentConverter
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding




DATA_PATH = Path("./test_data")
CLASS_NAME = "document_chunk_embedding"
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")




def init_weaviate():
    client = weaviate.connect_to_local()
    
    if client.collections.exists(CLASS_NAME):
        client.collections.delete(CLASS_NAME)

    client.collections.create(
    name=CLASS_NAME,
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
    ],
    vector_config=Configure.Vectors.self_provided(),
    )

    return client


def ingestion(client):
    converter = DocumentConverter()
    parser = MarkdownNodeParser()
    embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    collection = client.collections.use(CLASS_NAME)

    for file in DATA_PATH.glob("*.pdf"):
        print(f"Ingesting: {file.name}")

        markdown = converter.convert(file).document.export_to_markdown()
        nodes = parser.get_nodes_from_documents([Document(text=markdown)])

        for node in nodes:
            vector = embed_model.get_text_embedding(node.text)

            collection.data.insert(
                properties={
                    "text": node.text,
                    "source": file.name
                },
                vector=vector,
            )


def main():
    client = init_weaviate()
    try:
        ingestion(client)
    finally:
        client.close()

if __name__ == "__main__":
    main()