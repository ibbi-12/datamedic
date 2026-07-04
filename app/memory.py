"""Cross-run learning memory: distilled lessons from past failures.

After a run that crashed and then healed, the librarian LLM distills each
failure into a generalized lesson with trigger keywords. Before writing code
for a new dataset, lessons whose triggers appear in the CSV profile are
injected into the coder's prompt — so the same class of bug is avoided
instead of re-fixed.

Storage is a small JSON file (path via DATAMEDIC_LESSONS, default ./lessons.json).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

MAX_LESSONS = 50
MAX_RETRIEVED = 3

_lock = threading.Lock()


def _store_path() -> Path:
    default = Path(__file__).resolve().parent.parent / "lessons.json"
    return Path(os.environ.get("DATAMEDIC_LESSONS", str(default)))


def load_lessons() -> list[dict]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_lessons(new_lessons: list[dict]) -> int:
    """Append well-formed, non-duplicate lessons; returns how many were stored."""
    cleaned = [
        {
            "symptom": l["symptom"].strip(),
            "lesson": l["lesson"].strip(),
            "triggers": [t for t in l.get("triggers", []) if isinstance(t, str) and t.strip()],
        }
        for l in new_lessons
        if isinstance(l, dict) and l.get("symptom") and l.get("lesson")
    ]
    if not cleaned:
        return 0

    with _lock:
        existing = load_lessons()
        known = {l["symptom"].lower() for l in existing}
        added = []
        for lesson in cleaned:  # dedupe against the store AND within this batch
            key = lesson["symptom"].lower()
            if key not in known:
                known.add(key)
                added.append(lesson)
        if not added:
            return 0
        merged = (existing + added)[-MAX_LESSONS:]
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=1))
    return len(added)


def retrieve_lessons(csv_profile: str) -> list[dict]:
    """Lessons whose trigger keywords appear in the profile, best matches first.

    Distilled triggers don't always literally appear in a profile, so when nothing
    matches we fall back to the most recent lessons — cheap recency prior that keeps
    hard-won knowledge in play instead of stranding it on keyword mismatches.
    """
    lessons = load_lessons()
    profile_lower = csv_profile.lower()
    scored = []
    for lesson in lessons:
        hits = sum(1 for t in lesson.get("triggers", []) if t.lower() in profile_lower)
        if hits:
            scored.append((hits, lesson))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored:
        return [lesson for _, lesson in scored[:MAX_RETRIEVED]]
    return lessons[-2:][::-1]  # newest first


def format_lessons(lessons: list[dict]) -> str:
    """Render retrieved lessons as a prompt block."""
    lines = [f"- {l['symptom']}: {l['lesson']}" for l in lessons]
    return "\n".join(lines)
