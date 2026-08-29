"""
ai.memory — Per-session conversation memory for SatQuery AI.

Maintains query history and image references across turns within a session
so that the agent can refer back to earlier context.

Modules:
  session — SessionContext dataclass + SessionStore in-memory backend
"""
from __future__ import annotations
