"""Tests for Reciprocal Rank Fusion (RRF) and Parent-Document Retrieval."""
import pytest
from langchain_core.documents import Document
from app.retriever import reciprocal_rank_fusion, HybridParentRetriever


def test_rrf_fusion_logic():
    """
    Verifies that Reciprocal Rank Fusion correctly merges dense and sparse results.
    doc_b has rank 2 in dense and rank 1 in sparse -> RRF = 1/62 + 1/61 = 0.03252
    doc_a has rank 1 in dense and rank 3 in sparse -> RRF = 1/61 + 1/63 = 0.03226
    Therefore, doc_b must be ranked higher than doc_a.
    """
    dense_results = [
        {"id": "doc_a", "content": "Dense result A", "metadata": {"chunk_id": "c_a"}},
        {"id": "doc_b", "content": "Dense result B", "metadata": {"chunk_id": "c_b"}},
        {"id": "doc_c", "content": "Dense result C", "metadata": {"chunk_id": "c_c"}},
    ]

    sparse_results = [
        {"id": "doc_b", "content": "Sparse result B", "metadata": {"chunk_id": "c_b"}},
        {"id": "doc_d", "content": "Sparse result D", "metadata": {"chunk_id": "c_d"}},
        {"id": "doc_a", "content": "Sparse result A", "metadata": {"chunk_id": "c_a"}},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    # Assertions
    assert len(fused) == 4, "Fused results should combine all 4 unique document IDs"
    top_doc_id = fused[0]["id"]
    second_doc_id = fused[1]["id"]

    assert top_doc_id == "doc_b", f"Expected doc_b to rank 1st after RRF, got {top_doc_id}"
    assert second_doc_id == "doc_a", f"Expected doc_a to rank 2nd after RRF, got {second_doc_id}"

    # Verify RRF score ordering
    scores = [item["rrf_score"] for item in fused]
    assert scores == sorted(scores, reverse=True), "Fused results must be sorted descending by RRF score"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_parent_document_retrieval_mapping():
    """
    Verifies that child chunks are mapped back to their full parent documents.
    """
    retriever = HybridParentRetriever(
        parent_chunk_size=500,
        child_chunk_size=100,
        chunk_overlap=20
    )

    long_parent_content = (
        "PARENT HEADER: Introduction to distributed database consensus. "
        "Raft is a consensus algorithm designed as an alternative to Paxos. "
        "It was meant to be more understandable than Paxos by means of separation of logic. "
        "Child chunks should index small pieces of this text, but retrieval must return this entire parent."
    )

    parent_doc = Document(
        page_content=long_parent_content,
        metadata={"parent_id": "parent_db_consensus_001", "topic": "databases"}
    )

    retriever.add_parent_documents([parent_doc])

    # Search for child chunk keywords
    results = retriever.retrieve("Raft consensus algorithm", top_k=5)

    assert len(results) >= 1
    retrieved_parent = results[0]

    # Must return the full parent content, not just a 100-character chunk
    assert retrieved_parent.page_content == long_parent_content
    assert retrieved_parent.metadata["parent_id"] == "parent_db_consensus_001"
    assert "rrf_score" in retrieved_parent.metadata
    assert retrieved_parent.metadata["rrf_score"] > 0
