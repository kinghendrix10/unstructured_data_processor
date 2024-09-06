# setup.py
from setuptools import setup, find_packages

setup(
    name="unstructured_data_processor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "llama-index",
        "neo4j",
        "llama-index-graph-stores-neo4j",
        "llama-parse",
        "qdrant_client",
        "llama-index-vector-stores-qdrant",
        "llama-index-embeddings-huggingface",
        "llama-index-embeddings-fastembed",
        "llama-index-llms-groq",
        "python-docx"
    ],
    author="Huleji Tukura",
    description="A library for processing unstructured data and loading it into Neo4j",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kinghendrix10/unstructured_data_processor",
)
