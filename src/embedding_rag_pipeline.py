from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


def load_text_document(file_path: str) -> str:
    """
    Load a markdown or text document.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """
    Split text into overlapping text chunks.
    """

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


class ChromaRAGRetriever:
    """
    Embedding-based RAG retriever using SentenceTransformers + Chroma.

    This version supports:
    - document chunking
    - embedding generation
    - vector database storage
    - semantic retrieval
    """

    def __init__(
        self,
        collection_name: str = "business_context",
        persist_directory: str = "chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        reset_collection: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name

        if reset_collection and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        self.embedding_model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=persist_directory)

        existing_collections = [
            collection.name for collection in self.client.list_collections()
        ]

        if collection_name in existing_collections:
            self.collection = self.client.get_collection(name=collection_name)
        else:
            self.collection = self.client.create_collection(name=collection_name)

    def add_chunks(self, chunks: List[str]) -> None:
        """
        Add text chunks into Chroma collection.
        """

        if not chunks:
            raise ValueError("chunks cannot be empty")

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedding_model.encode(chunks).tolist()

        existing_count = self.collection.count()

        if existing_count > 0:
            return

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Retrieve semantically relevant chunks using vector similarity.
        """

        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved_chunks = []

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, distance in zip(ids, documents, distances):
            score = 1 / (1 + float(distance))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=document,
                    score=round(score, 3),
                )
            )

        return retrieved_chunks


def build_chroma_retriever_from_file(
    file_path: str,
    persist_directory: str = "chroma_db",
    reset_collection: bool = False,
) -> ChromaRAGRetriever:
    """
    Build a Chroma retriever from a text document.
    """

    text = load_text_document(file_path)
    chunks = chunk_text(text)

    retriever = ChromaRAGRetriever(
        persist_directory=persist_directory,
        reset_collection=reset_collection,
    )
    retriever.add_chunks(chunks)

    return retriever