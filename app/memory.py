"""Semantic Memory Manager for long-term user facts and persona adaptation."""
import json
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from app.config import settings


class SemanticMemoryManager:
    """
    Manages user profile facts and preferences across sessions (Mem0 / Zep pattern).
    Extracts facts from chat messages, persists them to local storage,
    and dynamically constructs persona adaptation prompts.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or settings.memory_storage_path
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _ensure_dir(self):
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load(self):
        """Loads memories from JSON storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._memories = json.load(f)
            except Exception:
                self._memories = {}
        else:
            self._memories = {}

    def _save(self):
        """Persists memories to JSON storage."""
        self._ensure_dir()
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save memories to {self.storage_path}: {e}")

    def extract_facts(self, message: str) -> List[str]:
        """
        Extracts user facts, preferred languages, and style choices from user messages.
        Uses semantic rule heuristics and regex patterns to ensure fast, zero-failure extraction.
        """
        extracted = []
        lower = message.lower()

        # Check for programming language preferences
        languages = ["python", "javascript", "typescript", "rust", "golang", "go", "java", "c\\+\\+", "c#"]
        for lang in languages:
            pattern = rf"\b(?:work in|code in|write in|develop in|use|prefer)\s+{lang}\b"
            if re.search(pattern, lower):
                clean_lang = "C++" if lang == "c\\+\\+" else lang.capitalize()
                extracted.append(f"User works in {clean_lang}")
                break

        # Check for brevity & code-only preferences
        if "code-only" in lower or "code only" in lower:
            extracted.append("User prefers brief, code-only responses")
        elif "brief" in lower or "concise" in lower:
            extracted.append("User prefers brief and concise responses")

        # Catch explicit 'prefer <X>' statements if not already captured
        if not any("prefers" in f for f in extracted):
            pref_match = re.search(r"\bprefer\s+([^.,;]+)", message, re.IGNORECASE)
            if pref_match:
                pref_text = pref_match.group(1).strip()
                fact = f"User prefers {pref_text}"
                if len(pref_text) > 3:
                    extracted.append(fact)

        return extracted

    def add_user_facts(self, user_id: str, facts: List[str]):
        """Adds extracted facts to the user's semantic memory profile."""
        if not user_id or not facts:
            return

        if user_id not in self._memories:
            self._memories[user_id] = []

        existing_texts = {m["fact"].lower() for m in self._memories[user_id]}
        now_iso = datetime.now(timezone.utc).isoformat()

        for fact in facts:
            clean_fact = fact.strip()
            if clean_fact and clean_fact.lower() not in existing_texts:
                self._memories[user_id].append({
                    "fact": clean_fact,
                    "created_at": now_iso
                })
                existing_texts.add(clean_fact.lower())

        self._save()

    def get_user_memories(self, user_id: str) -> List[str]:
        """Retrieves active semantic facts stored for a user."""
        records = self._memories.get(user_id, [])
        return [r["fact"] for r in records]

    def build_persona_prompt(self, user_id: str) -> str:
        """
        Constructs dynamic persona adaptation instructions for injection
        into the LLM's system prompt.
        """
        facts = self.get_user_memories(user_id)
        if not facts:
            return "You are an expert AI assistant that provides accurate and helpful answers based on retrieved context."

        facts_list = "\n".join(f"- {f}" for f in facts)
        adaptation_instructions = [
            "You are an expert AI assistant tailored to the user's specific persona.",
            "PERSISTENT USER PROFILE & PREFERENCES:",
            facts_list,
            "ADAPTATION RULES:",
            "1. Strictly enforce all user style, language, and brevity preferences listed above.",
            "2. If the user preferred code-only responses, return ONLY the raw code block with NO explanations or conversational filler.",
            "3. If the user preferred a specific programming language (e.g., Python), write all solutions exclusively in that language.",
            "4. Incorporate retrieved context when relevant."
        ]
        return "\n".join(adaptation_instructions)

    def clear(self):
        """Clears all stored memories (used in test fixtures)."""
        self._memories = {}
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception:
                pass


# Global singleton instance
memory_manager = SemanticMemoryManager()
