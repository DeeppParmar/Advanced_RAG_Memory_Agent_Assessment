# Advanced RAG Memory Agent
- **Repository**: https://github.com/DeeppParmar/Advanced_RAG_Memory_Agent_Assessment

Production-grade Advanced Retrieval-Augmented Generation (RAG) agent featuring dense-sparse hybrid search, parent-document retrieval mapping, Cohere ReRank context compression, long-term semantic memory (Mem0 pattern), and multi-phase execution latency telemetry.

---

## Installation

1. Configure `.env` file (`OPENAI_API_KEY`, `COHERE_API_KEY`, `QDRANT_URL`):
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run FastAPI:
   ```bash
   uvicorn main:app --reload
   ```
4. Run tests:
   ```bash
   pytest
   ```

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │     User / Client       │
                    └────────────┬────────────┘
                                 │ POST /api/chat
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Server                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Memory Fact Extraction & Persona Adaptation (Mem0)       │
│    - Extracts user facts (e.g. languages, brevity style)    │
│    - Loads user persona profile into LLM system prompt      │
├─────────────────────────────────────────────────────────────┤
│ 2. Hybrid Parent-Document Retriever                         │
│    - Dense Vectors (Qdrant semantic index)                  │
│    - Sparse Lexical (BM25Okapi keyword index)               │
│    - Reciprocal Rank Fusion (RRF: k=60)                     │
│    - Resolves winning child chunks (200 chars) to full      │
│      parent documents (1000 chars)                          │
├─────────────────────────────────────────────────────────────┤
│ 3. Cohere ReRank & Context Compression                      │
│    - Retrieves top 20 candidate parent documents            │
│    - Compresses to top 5 highest-scoring contexts           │
│    - Prunes chunks below score threshold (min_score: 0.3)   │
├─────────────────────────────────────────────────────────────┤
│ 4. Response Generation & Latency Telemetry Engine           │
│    - Tracks retrieval, rerank, generation, and total ms     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                     JSON Response Body
```

---

## Key Technical Components

### 1. Hybrid Search & Parent-Document Retriever
- **Parent-Document Mapping**: Splits long parent documents (~1000 characters) into smaller child segments (~200 characters) for high-precision semantic matching, then automatically resolves matched chunks back to their full parent context before LLM prompting.
- **Dense + Sparse Hybrid Index**:
  - Dense semantic search using Qdrant vector index.
  - Sparse lexical search using BM25Okapi.
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k = 60)$$
  Merges disparate scoring scales into a unified, fair rank list.

### 2. Cohere ReRank & Context Compression
- Compresses the initial 20 hybrid search candidates down to the top 5 highest-scoring documents using Cohere's `rerank-english-v3.0`.
- Implements a relevance score filter (`score_threshold = 0.3`) to purge low-relevance noise.

### 3. Long-Term Semantic Memory (Mem0 / Zep Pattern)
- Automatically extracts facts, preferred programming languages, and response styles from incoming chats.
- Persists user profiles across sessions in `data/user_memories.json`.
- Dynamically injects persona adaptation rules into system prompts for subsequent sessions.

### 4. Execution Latency Telemetry
Every `/api/chat` response includes millisecond-precision timing for benchmarking:
```json
{
  "retrieval_latency_ms": 1.15,
  "rerank_latency_ms": 8.42,
  "generation_latency_ms": 0.05,
  "total_latency_ms": 10.20
}
```

---

## API Endpoints Reference

### 1. Chat & Inference
- **Endpoint**: `POST /api/chat`
- **Request Body**:
  ```json
  {
    "query": "Write a quick merge sort routine.",
    "user_id": "usr_99",
    "session_id": "sess_001"
  }
  ```
- **Response**:
  ```json
  {
    "response": "```python\ndef merge_sort(arr):\n    ...\n```",
    "telemetry": {
      "retrieval_latency_ms": 1.15,
      "rerank_latency_ms": 8.42,
      "generation_latency_ms": 0.05,
      "total_latency_ms": 10.20
    },
    "context_documents": [
      {
        "content": "Merge Sort is an efficient, general-purpose...",
        "metadata": {
          "parent_id": "doc_merge_sort_001",
          "relevance_score": 0.95
        }
      }
    ]
  }
  ```

### 2. Semantic User Memory Retrieval
- **Endpoint**: `GET /api/users/{user_id}/memories`
- **Response**:
  ```json
  {
    "user_id": "usr_99",
    "memories": [
      "User works in Python",
      "User prefers brief, code-only responses"
    ],
    "count": 2
  }
  ```

### 3. Service Health
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "vector_store": "qdrant",
    "compressor": "cohere-rerank",
    "memory_store": "mem0-semantic"
  }
  ```

---

## Verification & Validation Workflows

Run the automated verification suite using `pytest`:
```bash
pytest -v
```

### Verification Test Cases Covered:
1. **Test Case 1: Hybrid Retrieval & Re-ranking Latency Validation**
   - Asserts `/api/chat` responds with valid positive numeric values for `retrieval_latency_ms`, `rerank_latency_ms`, `generation_latency_ms`, and `total_latency_ms`.
2. **Test Case 2: Multi-Session Memory Persona Adaptation**
   - Query 1: `'Hello! I work in Python and prefer brief, code-only responses.'` for `user_id='usr_99'`.
   - Asserts `/api/users/usr_99/memories` returns extracted Python preference facts.
   - Query 2 (new session): `'Write a quick merge sort routine.'` verifies the agent bypasses long conversational explanations and outputs only the raw Python code block.
3. **Test Case 3: Automated Test Suite Execution**
   - Unit tests for RRF fusion logic and parent mapping (`tests/test_rrf_retrieval.py`).
   - Mocked Cohere API score threshold filtering (`tests/test_rerank_filter.py`).
   - Integration tests with FastAPI test client (`tests/test_api.py`).

---

## API Collection Export

A complete Postman collection is included in [`postman_collection.json`](./postman_collection.json), containing pre-configured requests for all endpoints and validation scenarios.
