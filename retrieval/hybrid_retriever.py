"""
Hybrid Retrieval: FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion
Embeddings: sentence-transformers (free, local, no API needed)
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class FAISSRetriever:
    """Dense vector retrieval using FAISS + local sentence-transformers embeddings."""

    def __init__(self, embedding_model: str = None):
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"Loading embedding model: {model_name} (local, no API needed)...")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index = None
        self.documents: List[Document] = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.dimension}")

    async def get_embedding(self, text: str) -> List[float]:
        # sentence-transformers is sync — wrap it directly (fast enough)
        embedding = self.model.encode(text[:2000], normalize_embeddings=True)
        return embedding.tolist()

    def get_embedding_sync(self, text: str) -> np.ndarray:
        return self.model.encode(text[:2000], normalize_embeddings=True)

    async def build_index(self, documents: List[Document]):
        """Build FAISS index from documents using batch encoding (fast)."""
        self.documents = documents
        print(f"Building FAISS index for {len(documents)} documents (local embeddings)...")

        texts = [doc.content[:2000] for doc in documents]
        # Batch encode — sentence-transformers handles this efficiently
        embedding_matrix = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype(np.float32)

        # IndexFlatIP = cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embedding_matrix)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    async def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for top-k documents by dense similarity."""
        if self.index is None:
            return []

        query_vec = self.get_embedding_sync(query).reshape(1, -1).astype(np.float32)
        # Already normalized by sentence-transformers (normalize_embeddings=True)

        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.documents[idx]
                doc.score = float(score)
                results.append((doc, float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)


class BM25Retriever:
    """Sparse BM25 retrieval."""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: List[Document] = []

    def build_index(self, documents: List[Document]):
        """Build BM25 index from documents."""
        self.documents = documents
        tokenized_corpus = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built for {len(documents)} documents.")

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for top-k documents by BM25 score."""
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                doc.score = float(scores[idx])
                results.append((doc, float(scores[idx])))
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]


class HybridRetriever:
    """
    Combines FAISS dense + BM25 sparse retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        self.faiss_retriever = FAISSRetriever()
        self.bm25_retriever = BM25Retriever()
        self.rrf_k = rrf_k or int(os.getenv("RRF_K", 60))

    async def build_indexes(self, documents: List[Document]):
        """Build both FAISS and BM25 indexes."""
        self.bm25_retriever.build_index(documents)
        await self.faiss_retriever.build_index(documents)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k: int = None
    ) -> List[Document]:
        """
        Merge ranked lists using Reciprocal Rank Fusion.
        RRF score = sum(1 / (rank_i + K)) for each retriever
        """
        rrf_k = k or self.rrf_k
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Dense rankings
        for rank, (doc, _) in enumerate(dense_results, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rank + rrf_k)
            doc_map[doc.id] = doc

        # Sparse rankings
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rank + rrf_k)
            doc_map[doc.id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused_docs = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc.score = scores[doc_id]
            fused_docs.append(doc)
        return fused_docs

    async def search(self, query: str, k: int = 10) -> List[Document]:
        """Run hybrid search and return RRF-fused results."""
        fetch_k = k * 2  # Fetch more from each for better fusion

        dense_results = await self.faiss_retriever.search(query, k=fetch_k)
        sparse_results = self.bm25_retriever.search(query, k=fetch_k)

        fused = self.reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:k]

    def save(self, faiss_path: str, bm25_path: str):
        self.faiss_retriever.save(faiss_path)
        self.bm25_retriever.save(bm25_path)

    def load(self, faiss_path: str, bm25_path: str):
        self.faiss_retriever.load(faiss_path)
        self.bm25_retriever.load(bm25_path)
