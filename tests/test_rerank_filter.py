"""Tests for Cohere ReRank Context Compression and Score Threshold Filtering."""
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.reranker import FilteredCohereRerank


class MockRerankItem:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class MockCohereResponse:
    def __init__(self, results):
        self.results = results


def test_rerank_score_filtering_drops_low_scores():
    """
    Verifies that chunks falling below the minimum relevance score threshold
    are dropped by the compressor.
    Threshold: 0.3
    Candidate scores:
      doc 0: 0.92 (Keep)
      doc 1: 0.15 (Drop: < 0.3)
      doc 2: 0.81 (Keep)
      doc 3: 0.28 (Drop: < 0.3)
      doc 4: 0.74 (Keep)
    """
    compressor = FilteredCohereRerank(
        top_n=5,
        score_threshold=0.3,
        cohere_api_key="mock-key"
    )

    # Mock Cohere client response
    mock_client = MagicMock()
    mock_client.rerank.return_value = MockCohereResponse([
        MockRerankItem(index=0, relevance_score=0.92),
        MockRerankItem(index=1, relevance_score=0.15),
        MockRerankItem(index=2, relevance_score=0.81),
        MockRerankItem(index=3, relevance_score=0.28),
        MockRerankItem(index=4, relevance_score=0.74),
    ])
    compressor.client = mock_client

    input_docs = [
        Document(page_content="High relevance doc 0", metadata={"doc_id": 0}),
        Document(page_content="Irrelevant noise doc 1", metadata={"doc_id": 1}),
        Document(page_content="High relevance doc 2", metadata={"doc_id": 2}),
        Document(page_content="Low score noise doc 3", metadata={"doc_id": 3}),
        Document(page_content="Moderate relevance doc 4", metadata={"doc_id": 4}),
    ]

    compressed_docs = compressor.compress_documents(
        documents=input_docs,
        query="Test query for scoring"
    )

    # Verify that the two low-scoring documents (<0.3) were dropped
    assert len(compressed_docs) == 3, f"Expected 3 retained documents, got {len(compressed_docs)}"

    retained_ids = [doc.metadata["doc_id"] for doc in compressed_docs]
    assert retained_ids == [0, 2, 4], f"Expected doc IDs [0, 2, 4], got {retained_ids}"

    # Verify all retained documents have relevance_score >= 0.3
    for doc in compressed_docs:
        score = doc.metadata.get("relevance_score")
        assert score is not None
        assert score >= 0.3, f"Document score {score} is below threshold 0.3"


def test_rerank_top_n_compression_limit():
    """
    Verifies that even if more documents meet the threshold,
    the output is compressed to top_n highest scoring documents.
    """
    compressor = FilteredCohereRerank(
        top_n=2,
        score_threshold=0.2,
        cohere_api_key="mock-key"
    )

    mock_client = MagicMock()
    mock_client.rerank.return_value = MockCohereResponse([
        MockRerankItem(index=0, relevance_score=0.95),
        MockRerankItem(index=1, relevance_score=0.85),
        MockRerankItem(index=2, relevance_score=0.75),
        MockRerankItem(index=3, relevance_score=0.65),
    ])
    compressor.client = mock_client

    input_docs = [
        Document(page_content="Doc 0", metadata={"id": 0}),
        Document(page_content="Doc 1", metadata={"id": 1}),
        Document(page_content="Doc 2", metadata={"id": 2}),
        Document(page_content="Doc 3", metadata={"id": 3}),
    ]

    compressed_docs = compressor.compress_documents(input_docs, "query")

    # Should only return top_n = 2 documents
    assert len(compressed_docs) == 2
    assert compressed_docs[0].metadata["id"] == 0
    assert compressed_docs[1].metadata["id"] == 1
