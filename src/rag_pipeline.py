from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: int
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
    Split text into overlapping chunks.

    This is a simple character-based chunking method for MVP.
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


class SimpleRAGRetriever:
    """
    A lightweight local retriever using TF-IDF.

    This approximates vector search for the MVP.
    Later, this can be replaced with embeddings + Chroma/FAISS.
    """

    def __init__(self, chunks: List[str]):
        if not chunks:
            raise ValueError("chunks cannot be empty")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = [
            RetrievedChunk(
                chunk_id=int(index),
                text=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in top_indices
        ]

        return results


def build_retriever_from_file(file_path: str) -> SimpleRAGRetriever:
    text = load_text_document(file_path)
    chunks = chunk_text(text)
    return SimpleRAGRetriever(chunks)