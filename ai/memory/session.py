"""
Session memory management for multi-turn conversational analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class SessionContext:
    """State for a single user conversation session."""
    session_id: str
    query_history: List[str] = field(default_factory=list)
    image_refs: List[str] = field(default_factory=list)
    last_tool_results: List[Dict[str, Any]] = field(default_factory=list)
    cached_geojson: Optional[Dict[str, Any]] = None
    turn_count: int = 0

    def record_turn(self, query: str, image_filenames: List[str],
                    tool_results: Optional[List[Dict[str, Any]]] = None,
                    geojson: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed query turn."""
        self.query_history.append(query)
        self.image_refs.extend(image_filenames)
        if tool_results is not None:
            self.last_tool_results = tool_results
        if geojson is not None:
            self.cached_geojson = geojson
        self.turn_count += 1


class SessionStore:
    """In-memory multi-session store."""
    _store: Dict[str, SessionContext] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> SessionContext:
        """Retrieve an existing session or initialize a new one."""
        if session_id not in cls._store:
            cls._store[session_id] = SessionContext(session_id=session_id)
        return cls._store[session_id]

    @classmethod
    def update(cls, session_id: str, ctx: SessionContext) -> None:
        """Persist updated session context."""
        cls._store[session_id] = ctx

    @classmethod
    def delete(cls, session_id: str) -> None:
        """Remove a session."""
        cls._store.pop(session_id, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all active sessions (for test isolation)."""
        cls._store.clear()
