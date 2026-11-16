"""Integration tests for FastAPI endpoints, latency telemetry, and persona adaptation."""
import pytest
from fastapi.testclient import TestClient
from main import app
from app.memory import memory_manager


@pytest.fixture(autouse=True)
def clean_memory_fixture():
    """Ensure clean memory state before and after each test."""
    memory_manager.clear()
    yield
    memory_manager.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_case_1_hybrid_retrieval_and_reranking_latency_validation(client):
    """
    Test Case 1: Hybrid Retrieval & Re-ranking Latency Validation
    1. Start the FastAPI application server.
    2. Send a query payload to `/api/chat`. Verify that the JSON response contains the telemetry block.
    3. Assert that retrieval_latency_ms, rerank_latency_ms, and total_latency_ms contain valid positive numeric values.
    """
    payload = {
        "query": "Explain how merge sort works and its divide-and-conquer strategy.",
        "user_id": "usr_test_latency",
        "session_id": "sess_latency_01"
    }

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()
    assert "response" in data
    assert "telemetry" in data

    telemetry = data["telemetry"]
    assert "retrieval_latency_ms" in telemetry
    assert "rerank_latency_ms" in telemetry
    assert "generation_latency_ms" in telemetry
    assert "total_latency_ms" in telemetry

    # Validate positive numeric values
    retrieval_ms = telemetry["retrieval_latency_ms"]
    rerank_ms = telemetry["rerank_latency_ms"]
    generation_ms = telemetry["generation_latency_ms"]
    total_ms = telemetry["total_latency_ms"]

    assert isinstance(retrieval_ms, (int, float)) and retrieval_ms > 0, "retrieval_latency_ms must be > 0"
    assert isinstance(rerank_ms, (int, float)) and rerank_ms > 0, "rerank_latency_ms must be > 0"
    assert isinstance(generation_ms, (int, float)) and generation_ms > 0, "generation_latency_ms must be > 0"
    assert isinstance(total_ms, (int, float)) and total_ms > 0, "total_latency_ms must be > 0"

    # Total latency should encompass sub-stage latencies
    assert total_ms >= max(retrieval_ms, rerank_ms, generation_ms)


def test_case_2_multi_session_memory_persona_adaptation(client):
    """
    Test Case 2: Multi-Session Memory Persona Adaptation
    1. Send query 1: 'Hello! I work in Python and prefer brief, code-only responses.' using user_id='usr_99'.
    2. Call the memory retrieval endpoint: `/api/users/usr_99/memories`. Verify that the response lists the extracted Python preference fact.
    3. Open a new chat session. Send query 2: 'Write a quick merge sort routine.' Verify the agent bypasses long explanations and returns only the python code block, adapting to the user's stored profile.
    """
    user_id = "usr_99"

    # Step 1: Send query 1
    query_1_payload = {
        "query": "Hello! I work in Python and prefer brief, code-only responses.",
        "user_id": user_id,
        "session_id": "session_init"
    }
    res1 = client.post("/api/chat", json=query_1_payload)
    assert res1.status_code == 200

    # Step 2: Call memory retrieval endpoint
    memories_res = client.get(f"/api/users/{user_id}/memories")
    assert memories_res.status_code == 200

    mem_data = memories_res.json()
    assert mem_data["user_id"] == user_id
    assert "memories" in mem_data
    memories_list = mem_data["memories"]

    # Verify that the response lists the extracted Python preference fact
    assert any("python" in m.lower() for m in memories_list), f"Python fact missing from memories: {memories_list}"
    assert any("code-only" in m.lower() for m in memories_list), f"Code-only fact missing from memories: {memories_list}"

    # Step 3: Open a new chat session and send query 2
    query_2_payload = {
        "query": "Write a quick merge sort routine.",
        "user_id": user_id,
        "session_id": "session_new_tab"
    }
    res2 = client.post("/api/chat", json=query_2_payload)
    assert res2.status_code == 200

    answer = res2.json()["response"].strip()

    # Verify the agent bypasses long explanations and returns only the python code block
    assert answer.startswith("```python")
    assert answer.endswith("```")
    assert "def merge_sort" in answer
    # Verify absence of typical long conversational explanations
    assert "Here is" not in answer
    assert "Hope this helps" not in answer
    assert "In this explanation" not in answer


def test_api_empty_query_validation(client):
    """Verifies that an empty query string is rejected with HTTP 400."""
    response = client.post("/api/chat", json={
        "query": "   ",
        "user_id": "usr_test",
        "session_id": "s1"
    })
    assert response.status_code == 400


def test_health_check_endpoint(client):
    """Verifies /health status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
