"""Interactive Visual Testing Dashboard for the Advanced RAG Agent."""

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Advanced RAG Agent | Interactive Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #10b981;
      --code-bg: #090d16;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 12px;
    }
    h1 { font-size: 1.5rem; font-weight: 700; color: var(--primary); }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      font-size: 0.75rem;
      border-radius: 9999px;
      background: #0369a1;
      color: #e0f2fe;
      font-weight: 600;
    }
    .nav-links a {
      color: var(--primary);
      text-decoration: none;
      margin-left: 14px;
      font-size: 0.9rem;
      border: 1px solid var(--card-border);
      padding: 6px 12px;
      border-radius: 6px;
      transition: background 0.2s;
    }
    .nav-links a:hover { background: var(--card-bg); }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 20px;
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 12px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    p.desc { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px; }
    .btn {
      background: var(--primary);
      color: #0f172a;
      border: none;
      padding: 10px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.9rem;
      transition: all 0.2s;
      display: inline-block;
      margin-bottom: 8px;
      margin-right: 8px;
    }
    .btn:hover { background: var(--primary-hover); color: white; }
    .btn-secondary { background: #334155; color: white; }
    .btn-secondary:hover { background: #475569; }
    .telemetry-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 16px 0;
    }
    .metric-card {
      background: var(--code-bg);
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
      text-align: center;
    }
    .metric-val { font-size: 1.25rem; font-weight: 700; color: var(--accent); }
    .metric-lbl { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; }
    pre {
      background: var(--code-bg);
      color: #e2e8f0;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
      border: 1px solid var(--card-border);
      white-space: pre-wrap;
      max-height: 340px;
      overflow-y: auto;
    }
    .input-group { margin-bottom: 12px; }
    label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
    input, textarea {
      width: 100%;
      background: var(--code-bg);
      border: 1px solid var(--card-border);
      padding: 8px 12px;
      color: white;
      border-radius: 6px;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Advanced RAG Agent Dashboard</h1>
        <span class="badge">Hybrid Search (Dense+Sparse)</span>
        <span class="badge">Cohere ReRank</span>
        <span class="badge">Mem0 Semantic Memory</span>
      </div>
      <div class="nav-links">
        <a href="/docs" target="_blank">Swagger UI (/docs)</a>
        <a href="/health" target="_blank">Health Check (/health)</a>
      </div>
    </header>

    <div class="grid">
      <!-- Test Case 1 Card -->
      <div class="card">
        <div class="card-title">Test Case 1: Latency & Telemetry</div>
        <p class="desc">Executes hybrid dense-sparse search, Cohere ReRank compression, and asserts positive execution latency telemetry.</p>
        <button class="btn" onclick="runTestCase1()">Run Latency Test</button>
        <div class="telemetry-bar">
          <div class="metric-card"><div class="metric-val" id="tc1-retrieval">-</div><div class="metric-lbl">Retrieval ms</div></div>
          <div class="metric-card"><div class="metric-val" id="tc1-rerank">-</div><div class="metric-lbl">ReRank ms</div></div>
          <div class="metric-card"><div class="metric-val" id="tc1-gen">-</div><div class="metric-lbl">Generation ms</div></div>
          <div class="metric-card"><div class="metric-val" id="tc1-total">-</div><div class="metric-lbl">Total ms</div></div>
        </div>
        <pre id="tc1-output">// Output will appear here...</pre>
      </div>

      <!-- Test Case 2 Card -->
      <div class="card">
        <div class="card-title">Test Case 2: Multi-Session Memory Adaptation</div>
        <p class="desc">Tests long-term user fact extraction (Python + brief code-only) and cross-session persona response adaptation.</p>
        <div>
          <button class="btn" onclick="runTC2Step1()">1. Set Persona (usr_99)</button>
          <button class="btn btn-secondary" onclick="runTC2Step2()">2. View Memories</button>
          <button class="btn" onclick="runTC2Step3()">3. Query Merge Sort (New Session)</button>
        </div>
        <pre id="tc2-output">// Follow the 3 steps above to test multi-session memory adaptation...</pre>
      </div>
    </div>

    <!-- Interactive Chat Sandbox -->
    <div class="card">
      <div class="card-title">Interactive Custom Chat Sandbox</div>
      <p class="desc">Test any custom prompt against the RAG retrieval pipeline and memory engine.</p>
      <div style="display: flex; gap: 12px; margin-bottom: 12px;">
        <div class="input-group" style="flex: 1;">
          <label>User ID</label>
          <input type="text" id="custom-user" value="usr_test">
        </div>
        <div class="input-group" style="flex: 1;">
          <label>Session ID</label>
          <input type="text" id="custom-session" value="sess_web_ui">
        </div>
      </div>
      <div class="input-group">
        <label>Query / Prompt</label>
        <textarea id="custom-query" rows="2">Write a quick merge sort routine.</textarea>
      </div>
      <button class="btn" onclick="runCustomChat()">Send to /api/chat</button>
      
      <div class="telemetry-bar">
        <div class="metric-card"><div class="metric-val" id="custom-retrieval">-</div><div class="metric-lbl">Retrieval ms</div></div>
        <div class="metric-card"><div class="metric-val" id="custom-rerank">-</div><div class="metric-lbl">ReRank ms</div></div>
        <div class="metric-card"><div class="metric-val" id="custom-gen">-</div><div class="metric-lbl">Generation ms</div></div>
        <div class="metric-card"><div class="metric-val" id="custom-total">-</div><div class="metric-lbl">Total ms</div></div>
      </div>
      <pre id="custom-output">// API response output...</pre>
    </div>
  </div>

  <script>
    async function runTestCase1() {
      const out = document.getElementById("tc1-output");
      out.textContent = "Processing hybrid retrieval and Cohere ReRank compression...";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: "Explain how merge sort works and its divide-and-conquer strategy.",
            user_id: "usr_latency_test",
            session_id: "sess_latency_01"
          })
        });
        const data = await res.json();
        document.getElementById("tc1-retrieval").textContent = data.telemetry.retrieval_latency_ms;
        document.getElementById("tc1-rerank").textContent = data.telemetry.rerank_latency_ms;
        document.getElementById("tc1-gen").textContent = data.telemetry.generation_latency_ms;
        document.getElementById("tc1-total").textContent = data.telemetry.total_latency_ms;
        out.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = "Error: " + e.message;
      }
    }

    async function runTC2Step1() {
      const out = document.getElementById("tc2-output");
      out.textContent = "Sending preference registration query for usr_99...";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: "Hello! I work in Python and prefer brief, code-only responses.",
            user_id: "usr_99",
            session_id: "sess_init_99"
          })
        });
        const data = await res.json();
        out.textContent = "Step 1 Completed! Response from agent:\n" + data.response;
      } catch (e) {
        out.textContent = "Error: " + e.message;
      }
    }

    async function runTC2Step2() {
      const out = document.getElementById("tc2-output");
      out.textContent = "Querying /api/users/usr_99/memories...";
      try {
        const res = await fetch("/api/users/usr_99/memories");
        const data = await res.json();
        out.textContent = "Step 2 Retrieved Semantic Memories for usr_99:\n" + JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = "Error: " + e.message;
      }
    }

    async function runTC2Step3() {
      const out = document.getElementById("tc2-output");
      out.textContent = "Sending 'Write a quick merge sort routine.' in a NEW session...";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: "Write a quick merge sort routine.",
            user_id: "usr_99",
            session_id: "sess_new_tab_" + Date.now()
          })
        });
        const data = await res.json();
        out.textContent = "Step 3 Result (Notice: Bypasses explanations, returns ONLY code!):\n\n" + data.response;
      } catch (e) {
        out.textContent = "Error: " + e.message;
      }
    }

    async function runCustomChat() {
      const out = document.getElementById("custom-output");
      out.textContent = "Processing...";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: document.getElementById("custom-query").value,
            user_id: document.getElementById("custom-user").value,
            session_id: document.getElementById("custom-session").value
          })
        });
        const data = await res.json();
        document.getElementById("custom-retrieval").textContent = data.telemetry.retrieval_latency_ms;
        document.getElementById("custom-rerank").textContent = data.telemetry.rerank_latency_ms;
        document.getElementById("custom-gen").textContent = data.telemetry.generation_latency_ms;
        document.getElementById("custom-total").textContent = data.telemetry.total_latency_ms;
        out.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = "Error: " + e.message;
      }
    }
  </script>
</body>
</html>
"""
