"""UI module serving the user-friendly visual dashboard from static/index.html."""
import os

STATIC_INDEX_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


def get_html_dashboard() -> str:
    """Reads and returns the HTML dashboard."""
    if os.path.exists(STATIC_INDEX_PATH):
        with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body><h1>Smart RAG Assistant</h1><p>Dashboard loaded.</p></body></html>"


HTML_DASHBOARD = get_html_dashboard()
