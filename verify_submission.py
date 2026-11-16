"""Formal verification script executing Section 4 test cases against live server."""
import urllib.request
import json
import sys
import subprocess

BASE_URL = "http://127.0.0.1:8000"


def post_json(endpoint: str, data: dict):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode("utf-8"))


def get_json(endpoint: str):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode("utf-8"))


def run_all_verifications():
    print("=" * 70)
    print("RUNNING SECTION 4: VERIFICATION & VALIDATION TEST CASES")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST CASE 1: Hybrid Retrieval & Re-ranking Latency Validation
    # ---------------------------------------------------------
    print("\n--- [TEST CASE 1: Hybrid Retrieval & Re-ranking Latency Validation] ---")
    query_payload = {
        "query": "Explain how merge sort works and its divide-and-conquer strategy.",
        "user_id": "usr_latency_validation",
        "session_id": "sess_tc1"
    }
    status, data = post_json("/api/chat", query_payload)
    print(f"1. Sent query payload to /api/chat -> Status: {status}")
    assert status == 200, f"Expected status 200, got {status}"

    print("2. Verifying telemetry block...")
    assert "telemetry" in data, "JSON response must contain 'telemetry' block"
    telemetry = data["telemetry"]
    print(f"   Telemetry Block: {telemetry}")

    print("3. Asserting latency metrics contain valid positive numeric values...")
    retrieval_ms = telemetry.get("retrieval_latency_ms")
    rerank_ms = telemetry.get("rerank_latency_ms")
    generation_ms = telemetry.get("generation_latency_ms")
    total_ms = telemetry.get("total_latency_ms")

    assert isinstance(retrieval_ms, (int, float)) and retrieval_ms > 0, f"Invalid retrieval_latency_ms: {retrieval_ms}"
    assert isinstance(rerank_ms, (int, float)) and rerank_ms > 0, f"Invalid rerank_latency_ms: {rerank_ms}"
    assert isinstance(generation_ms, (int, float)) and generation_ms > 0, f"Invalid generation_latency_ms: {generation_ms}"
    assert isinstance(total_ms, (int, float)) and total_ms > 0, f"Invalid total_latency_ms: {total_ms}"

    print(f"   retrieval_latency_ms = {retrieval_ms} ms (> 0)  -> PASS")
    print(f"   rerank_latency_ms    = {rerank_ms} ms (> 0)     -> PASS")
    print(f"   generation_latency_ms= {generation_ms} ms (> 0) -> PASS")
    print(f"   total_latency_ms     = {total_ms} ms (> 0)      -> PASS")
    print(">>> TEST CASE 1: PASSED SUCCESSFULLY!")

    # ---------------------------------------------------------
    # TEST CASE 2: Multi-Session Memory Persona Adaptation
    # ---------------------------------------------------------
    print("\n--- [TEST CASE 2: Multi-Session Memory Persona Adaptation] ---")
    
    # Step 1
    print("1. Sending query 1: 'Hello! I work in Python and prefer brief, code-only responses.' using user_id='usr_99'...")
    q1_payload = {
        "query": "Hello! I work in Python and prefer brief, code-only responses.",
        "user_id": "usr_99",
        "session_id": "sess_initial"
    }
    status1, res1 = post_json("/api/chat", q1_payload)
    assert status1 == 200, f"Step 1 failed with status {status1}"
    print(f"   Agent response to Query 1: {res1['response']}")

    # Step 2
    print("2. Calling memory retrieval endpoint: /api/users/usr_99/memories...")
    status2, mem_data = get_json("/api/users/usr_99/memories")
    assert status2 == 200, f"Step 2 failed with status {status2}"
    memories = mem_data.get("memories", [])
    print(f"   Retrieved memories: {memories}")
    assert any("python" in m.lower() for m in memories), f"Extracted Python preference fact missing: {memories}"
    assert any("code-only" in m.lower() for m in memories), f"Extracted code-only preference fact missing: {memories}"
    print("   Extracted Python preference fact verified -> PASS")

    # Step 3
    print("3. Opening a new chat session ('sess_new_tab'). Sending query 2: 'Write a quick merge sort routine.'...")
    q2_payload = {
        "query": "Write a quick merge sort routine.",
        "user_id": "usr_99",
        "session_id": "sess_new_tab"
    }
    status3, res2 = post_json("/api/chat", q2_payload)
    assert status3 == 200, f"Step 3 failed with status {status3}"
    answer = res2["response"].strip()
    print("   Agent Response in New Session:\n" + "-" * 40 + "\n" + answer + "\n" + "-" * 40)

    # Verification: Bypasses long explanations and returns only the python code block
    assert "```python" in answer, "Response must contain python code block"
    assert "def merge_sort" in answer, "Response must contain merge sort implementation"
    
    # Check that conversational filler is bypassed
    filler_phrases = ["here is", "sure!", "i have written", "hope this helps", "in this explanation"]
    for filler in filler_phrases:
        assert filler not in answer.lower(), f"Response contains conversational filler: '{filler}'"

    print("   Verified: Bypassed long explanations and returned only python code block -> PASS")
    print(">>> TEST CASE 2: PASSED SUCCESSFULLY!")

    # ---------------------------------------------------------
    # TEST CASE 3: Automated Test Suite Execution
    # ---------------------------------------------------------
    print("\n--- [TEST CASE 3: Automated Test Suite Execution] ---")
    print("1. Running test suite using pytest...")
    pytest_res = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        capture_output=True,
        text=True
    )
    print(pytest_res.stdout)
    assert pytest_res.returncode == 0, f"Pytest failed with exit code {pytest_res.returncode}"
    print(">>> TEST CASE 3: ALL TESTS PASSED WITH NO EXCEPTIONS!")

    print("\n" + "=" * 70)
    print("ALL 3 VERIFICATION & VALIDATION TEST CASES PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_verifications()
