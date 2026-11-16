"""Ultra-minimal, clean visual testing UI with pre-filled examples."""

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Agent - Minimal UI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      padding: 32px 16px;
      display: flex;
      justify-content: center;
    }
    .app {
      width: 100%;
      max-width: 860px;
    }
    header {
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 16px;
    }
    h1 { font-size: 1.25rem; font-weight: 600; color: #38bdf8; }
    .links a {
      color: #94a3b8;
      text-decoration: none;
      font-size: 0.85rem;
      margin-left: 12px;
      padding: 4px 8px;
      border: 1px solid #334155;
      border-radius: 4px;
    }
    .links a:hover { color: #38bdf8; border-color: #38bdf8; }
    
    .panel {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .panel-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #cbd5e1;
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .examples-bar {
      margin-bottom: 16px;
    }
    .example-tag {
      display: inline-block;
      background: #090d16;
      border: 1px solid #334155;
      color: #38bdf8;
      font-size: 0.78rem;
      padding: 6px 12px;
      border-radius: 20px;
      cursor: pointer;
      margin: 0 6px 6px 0;
      transition: all 0.15s ease;
    }
    .example-tag:hover {
      background: #38bdf8;
      color: #0f172a;
      border-color: #38bdf8;
    }
    
    .form-row {
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }
    .field { flex: 1; }
    label {
      display: block;
      font-size: 0.75rem;
      color: #94a3b8;
      margin-bottom: 4px;
      text-transform: uppercase;
    }
    input, textarea {
      width: 100%;
      background: #090d16;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 10px 12px;
      color: #f8fafc;
      font-size: 0.9rem;
      outline: none;
      font-family: inherit;
    }
    input:focus, textarea:focus { border-color: #38bdf8; }
    
    .btn-group {
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }
    .btn {
      background: #38bdf8;
      color: #0f172a;
      font-weight: 600;
      border: none;
      border-radius: 6px;
      padding: 10px 18px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .btn-secondary {
      background: #334155;
      color: #f1f5f9;
    }
    .btn-secondary:hover { background: #475569; }

    .telemetry {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 16px 0;
    }
    .telemetry-item {
      background: #090d16;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 10px;
      text-align: center;
    }
    .telemetry-val {
      font-size: 1.15rem;
      font-weight: 700;
      color: #34d399;
    }
    .telemetry-lbl {
      font-size: 0.65rem;
      color: #94a3b8;
      text-transform: uppercase;
      margin-top: 2px;
    }

    pre {
      background: #090d16;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 14px;
      font-size: 0.85rem;
      color: #e2e8f0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 400px;
      overflow-y: auto;
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Advanced RAG Agent</h1>
      <div class="links">
        <a href="/docs" target="_blank">Swagger API</a>
        <a href="/health" target="_blank">Health</a>
      </div>
    </header>

    <div class="panel">
      <div class="panel-title">Click An Example to Auto-Fill</div>
      <div class="examples-bar">
        <span class="example-tag" onclick="fillExample(1)">Ex 1: Latency Benchmark (Merge Sort)</span>
        <span class="example-tag" onclick="fillExample(2)">Ex 2: Set Persona (usr_99 Python brief)</span>
        <span class="example-tag" onclick="fillExample(3)">Ex 3: Persona Adapted Query (Merge Sort)</span>
        <span class="example-tag" onclick="fillExample(4)">Ex 4: Advanced RAG Architecture</span>
      </div>

      <div class="form-row">
        <div class="field">
          <label>User ID</label>
          <input type="text" id="userId" value="usr_99" placeholder="e.g. usr_99">
        </div>
        <div class="field">
          <label>Session ID</label>
          <input type="text" id="sessionId" value="session_01" placeholder="e.g. session_01">
        </div>
      </div>

      <div class="field">
        <label>Query / Prompt</label>
        <textarea id="queryText" rows="3" placeholder="Enter your query here...">Explain how merge sort works and its divide-and-conquer strategy.</textarea>
      </div>

      <div class="btn-group">
        <button class="btn" onclick="sendChat()">Send Query (/api/chat)</button>
        <button class="btn btn-secondary" onclick="checkMemories()">View Stored Memories</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Execution Latency Telemetry</div>
      <div class="telemetry">
        <div class="telemetry-item">
          <div class="telemetry-val" id="valRetrieval">-</div>
          <div class="telemetry-lbl">Retrieval ms</div>
        </div>
        <div class="telemetry-item">
          <div class="telemetry-val" id="valRerank">-</div>
          <div class="telemetry-lbl">ReRank ms</div>
        </div>
        <div class="telemetry-item">
          <div class="telemetry-val" id="valGen">-</div>
          <div class="telemetry-lbl">Generation ms</div>
        </div>
        <div class="telemetry-item">
          <div class="telemetry-val" id="valTotal">-</div>
          <div class="telemetry-lbl">Total ms</div>
        </div>
      </div>

      <div class="panel-title" style="margin-top: 16px;">Response Output</div>
      <pre id="outputView">// Click an example above and click "Send Query" to see output...</pre>
    </div>
  </div>

  <script>
    const examples = {
      1: {
        user: "usr_latency",
        session: "sess_01",
        query: "Explain how merge sort works and its divide-and-conquer strategy."
      },
      2: {
        user: "usr_99",
        session: "sess_init",
        query: "Hello! I work in Python and prefer brief, code-only responses."
      },
      3: {
        user: "usr_99",
        session: "sess_new_tab",
        query: "Write a quick merge sort routine."
      },
      4: {
        user: "usr_architect",
        session: "sess_rag",
        query: "How does Parent-Document retrieval differ from standard RAG?"
      }
    };

    function fillExample(id) {
      const ex = examples[id];
      document.getElementById("userId").value = ex.user;
      document.getElementById("sessionId").value = ex.session;
      document.getElementById("queryText").value = ex.query;
    }

    async function sendChat() {
      const out = document.getElementById("outputView");
      out.textContent = "Processing hybrid retrieval, Cohere ReRank, and response generation...";

      const user_id = document.getElementById("userId").value;
      const session_id = document.getElementById("sessionId").value;
      const query = document.getElementById("queryText").value;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, user_id, session_id })
        });
        const data = await res.json();

        if (data.telemetry) {
          document.getElementById("valRetrieval").textContent = data.telemetry.retrieval_latency_ms;
          document.getElementById("valRerank").textContent = data.telemetry.rerank_latency_ms;
          document.getElementById("valGen").textContent = data.telemetry.generation_latency_ms;
          document.getElementById("valTotal").textContent = data.telemetry.total_latency_ms;
        }

        out.textContent = data.response + "\\n\\n--- FULL TELEMETRY & CONTEXT ---\\n" + JSON.stringify(data, null, 2);
      } catch (err) {
        out.textContent = "Error: " + err.message;
      }
    }

    async function checkMemories() {
      const user_id = document.getElementById("userId").value;
      const out = document.getElementById("outputView");
      out.textContent = "Fetching memories for user: " + user_id + "...";

      try {
        const res = await fetch("/api/users/" + encodeURIComponent(user_id) + "/memories");
        const data = await res.json();
        out.textContent = "STORED MEMORIES FOR " + user_id + ":\\n\\n" + JSON.stringify(data, null, 2);
      } catch (err) {
        out.textContent = "Error: " + err.message;
      }
    }
  </script>
</body>
</html>
"""
