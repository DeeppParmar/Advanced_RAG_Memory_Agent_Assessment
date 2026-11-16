"""Hybrid Search and Parent-Document Retrieval Pipeline with Reciprocal Rank Fusion (RRF)."""
from typing import List, Dict, Any, Optional
import uuid
import math
import re
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse search results using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1.0 / (k + rank)) for each rank list.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = str(doc.get("id") or doc.get("metadata", {}).get("chunk_id", rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc

    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = str(doc.get("id") or doc.get("metadata", {}).get("chunk_id", rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc

    # Sort descending by fused RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_results = []
    for doc_id in sorted_ids:
        item = dict(doc_lookup[doc_id])
        item["rrf_score"] = scores[doc_id]
        fused_results.append(item)

    return fused_results


def _tokenize(text: str) -> List[str]:
    """Simple lowercase regex tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _simple_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Deterministic normalized embedding generator for local/mock/testing environments
    when live OpenAI/Cohere API keys are not provided.
    """
    vec = [0.0] * dim
    words = _tokenize(text)
    if not words:
        return [1.0 / math.sqrt(dim)] * dim
    for w in words:
        h = hash(w)
        for i in range(dim):
            bit = (h >> (i % 31)) & 1
            vec[i] += 1.0 if bit == 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HybridParentRetriever:
    """
    Combines Parent-Document retrieval mapping with Dense (Qdrant) and
    Sparse (BM25) search indices via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        parent_chunk_size: int = settings.parent_chunk_size,
        child_chunk_size: int = settings.child_chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        rrf_k: int = settings.rrf_k
    ):
        self.collection_name = collection_name or settings.qdrant_collection
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k

        # Initialize Qdrant Client (in-memory by default)
        if qdrant_client:
            self.qdrant = qdrant_client
        elif settings.qdrant_url == ":memory:":
            self.qdrant = QdrantClient(location=":memory:")
        else:
            self.qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )

        # Parent Document Store: parent_id -> Document
        self.parent_doc_store: Dict[str, Document] = {}

        # Child chunks for BM25 and tracking
        self.child_docs: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dim = 64

        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        """Creates collection in Qdrant if it does not exist."""
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dim,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception:
            # Collection may already exist or running in mock
            pass

    def add_parent_documents(self, documents: List[Document]):
        """
        Splits large parent documents into parent nodes and smaller child chunks,
        indexes child chunks in Qdrant and BM25, and maintains parent document mapping.
        """
        for doc in documents:
            content = doc.page_content
            # Split into parent nodes if necessary, or keep as is if within parent_chunk_size
            parent_id = doc.metadata.get("parent_id") or str(uuid.uuid4())
            parent_doc = Document(
                page_content=content,
                metadata={**doc.metadata, "parent_id": parent_id}
            )
            self.parent_doc_store[parent_id] = parent_doc

            # Split parent into smaller child chunks (e.g. 200 chars)
            step = max(1, self.child_chunk_size - self.chunk_overlap)
            child_idx = 0
            for start in range(0, len(content), step):
                chunk_text = content[start:start + self.child_chunk_size].strip()
                if not chunk_text:
                    continue
                child_id = f"{parent_id}_chunk_{child_idx}"
                child_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "chunk_index": child_idx
                    }
                )
                self.child_docs.append(child_doc)
                child_idx += 1

                # Index in Qdrant
                vec = _simple_embedding(chunk_text, self.dim)
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qmodels.PointStruct(
                            id=abs(hash(child_id)) % (10**12),
                            vector=vec,
                            payload={
                                "child_id": child_id,
                                "parent_id": parent_id,
                                "content": chunk_text,
                                "metadata": child_doc.metadata
                            }
                        )
                    ]
                )

        # Re-build BM25 index on child corpus
        if self.child_docs:
            corpus = [_tokenize(d.page_content) for d in self.child_docs]
            self.bm25 = BM25Okapi(corpus)

    def search_dense(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Dense semantic vector search against Qdrant child chunk index."""
        if not self.child_docs:
            return []
        query_vec = _simple_embedding(query, self.dim)
        try:
            if hasattr(self.qdrant, "query_points"):
                response = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k
                )
                hits = response.points
            elif hasattr(self.qdrant, "search"):
                hits = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=top_k
                )
            else:
                hits = []
        except Exception:
            hits = []

        results = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            score = getattr(hit, "score", 0.0)
            results.append({
                "id": payload.get("child_id"),
                "parent_id": payload.get("parent_id"),
                "content": payload.get("content", ""),
                "score": score,
                "metadata": payload.get("metadata", {})
            })
        return results

    def search_sparse(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Sparse BM25 lexical keyword search against child chunk index."""
        if not self.bm25 or not self.child_docs:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        bm25_scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            doc = self.child_docs[idx]
            results.append({
                "id": doc.metadata.get("child_id"),
                "parent_id": doc.metadata.get("parent_id"),
                "content": doc.page_content,
                "score": float(bm25_scores[idx]),
                "metadata": doc.metadata
            })
        return results

    def retrieve(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Executes hybrid dense-sparse search, fuses candidate child chunks via RRF,
        and resolves winning child chunks to full parent documents.
        """
        dense_results = self.search_dense(query, top_k=top_k)
        sparse_results = self.search_sparse(query, top_k=top_k)

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self.rrf_k)

        # Resolve child chunks to unique parent documents
        seen_parents = set()
        parent_documents: List[Document] = []

        for item in fused:
            parent_id = item.get("parent_id") or item.get("metadata", {}).get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_doc = self.parent_doc_store.get(parent_id)
                if parent_doc:
                    # Attach retrieval telemetry metadata to document
                    doc_copy = Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **parent_doc.metadata,
                            "rrf_score": item.get("rrf_score", 0.0),
                            "matched_child_chunk": item.get("content", "")
                        }
                    )
                    parent_documents.append(doc_copy)

        return parent_documents[:top_k]
