"""
Lightweight Analysis Session Store for SatQuery AI.
Stores and retrieves analysis sessions, conversation threads, stable evidence,
and cached raster imagery across multi-turn interactions.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .rasterio_utils import RasterInput
except ImportError:
    from rasterio_utils import RasterInput


class SessionStore:
    """Thread-safe persistent session store for SatQuery AI multi-turn analyses."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            self.storage_dir = Path(__file__).parent / "data" / "sessions"
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem_cache: Dict[str, Dict[str, Any]] = {}
        self._raster_cache: Dict[str, List[RasterInput]] = {}

    def _session_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _raster_dir(self, session_id: str) -> Path:
        d = self.storage_dir / session_id / "rasters"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_session(
        self,
        session_id: str,
        initial_query: str,
        rasters: List[RasterInput],
        analysis: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and persist a new Analysis Session with image rasters."""
        now = datetime.now(timezone.utc).isoformat()

        # Save raster files to disk for future specialist tool runs
        raster_dir = self._raster_dir(session_id)
        saved_raster_paths = []
        image_filenames = []
        modalities = []

        for r in rasters:
            fn = r.filename or "image.png"
            image_filenames.append(fn)
            modalities.append(getattr(r, "modality", "optical"))
            target_path = raster_dir / fn
            try:
                target_path.write_bytes(r.data)
                saved_raster_paths.append(str(target_path))
            except Exception:
                pass

        primary_modality = modalities[0] if modalities else "optical"
        img_meta = rasters[0].metadata if rasters and hasattr(rasters[0], "metadata") else {}

        # Conversation history starts with the initial query and answer
        initial_answer = analysis.get("answer", "")
        conversation = [
            {
                "role": "user",
                "content": initial_query,
                "timestamp": now,
            },
            {
                "role": "assistant",
                "content": initial_answer,
                "timestamp": now,
                "claims": analysis.get("claims", []),
                "evidence_ids": [e.get("evidence_id") for e in evidence if e.get("evidence_id")],
            }
        ]

        session_data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "initial_query": initial_query,
            "image": {
                "image_id": f"img_{session_id[:8]}",
                "filenames": image_filenames,
                "modality": primary_modality,
                "modalities": modalities,
                "metadata": img_meta,
                "raster_paths": saved_raster_paths,
            },
            "analysis": analysis,
            "evidence": evidence,
            "tool_results": tool_results,
            "conversation": conversation,
            "execution_metadata": execution_metadata or {},
        }

        with self._lock:
            self._mem_cache[session_id] = session_data
            self._raster_cache[session_id] = rasters
            self._persist(session_id, session_data)

        return session_data

    def _persist(self, session_id: str, data: Dict[str, Any]) -> None:
        target_file = self._session_path(session_id)
        temp_file = target_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(target_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by session_id."""
        with self._lock:
            if session_id in self._mem_cache:
                return self._mem_cache[session_id]

            p = self._session_path(session_id)
            if not p.exists():
                return None

            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._mem_cache[session_id] = data
                return data
            except Exception:
                return None

    def get_session_rasters(self, session_id: str) -> List[RasterInput]:
        """Retrieve live RasterInput instances for a session (from cache or disk)."""
        with self._lock:
            if session_id in self._raster_cache:
                return self._raster_cache[session_id]

        sess = self.get_session(session_id)
        if not sess:
            return []

        rasters = []
        raster_paths = sess.get("image", {}).get("raster_paths", [])
        for p_str in raster_paths:
            p = Path(p_str)
            if p.exists():
                try:
                    data = p.read_bytes()
                    rasters.append(RasterInput(p.name, data))
                except Exception:
                    pass

        with self._lock:
            if rasters:
                self._raster_cache[session_id] = rasters
        return rasters

    def update_session(
        self,
        session_id: str,
        user_query: str,
        assistant_response: Dict[str, Any],
        new_evidence: Optional[List[Dict[str, Any]]] = None,
        new_tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing session with a new conversation turn and updated findings."""
        with self._lock:
            sess = self.get_session(session_id)
            if not sess:
                return None

            now = datetime.now(timezone.utc).isoformat()
            sess["updated_at"] = now

            # Append new user message
            sess["conversation"].append({
                "role": "user",
                "content": user_query,
                "timestamp": now,
            })

            # Append new assistant message
            claims = assistant_response.get("claims", [])
            sess["conversation"].append({
                "role": "assistant",
                "content": assistant_response.get("answer", ""),
                "timestamp": now,
                "claims": claims,
                "evidence_ids": [
                    eid for c in claims for eid in c.get("evidence_ids", [])
                ] if claims else [],
            })

            # Update latest structured analysis
            sess["analysis"] = {
                "answer": assistant_response.get("answer", ""),
                "sections": assistant_response.get("sections", []),
                "claims": assistant_response.get("claims", []),
                "uncertainties": assistant_response.get("uncertainties", []),
                "justification": assistant_response.get("justification", ""),
            }

            # Append new evidence items without renaming or deleting existing ones
            if new_evidence:
                existing_ids = {e.get("evidence_id") for e in sess.get("evidence", [])}
                for item in new_evidence:
                    if item.get("evidence_id") not in existing_ids:
                        sess["evidence"].append(item)
                        existing_ids.add(item.get("evidence_id"))

            # Append new tool results
            if new_tool_results:
                sess["tool_results"].extend(new_tool_results)

            if assistant_response.get("execution_details"):
                sess["execution_metadata"].update(assistant_response["execution_details"])

            self._mem_cache[session_id] = sess
            self._persist(session_id, sess)
            return sess

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its cached rasters."""
        with self._lock:
            self._mem_cache.pop(session_id, None)
            self._raster_cache.pop(session_id, None)

            p = self._session_path(session_id)
            deleted = False
            if p.exists():
                p.unlink(missing_ok=True)
                deleted = True

            raster_dir = self.storage_dir / session_id
            if raster_dir.exists():
                shutil.rmtree(raster_dir, ignore_errors=True)
            return deleted

    def clear(self) -> None:
        """Clear all sessions."""
        with self._lock:
            self._mem_cache.clear()
            self._raster_cache.clear()
            if self.storage_dir.exists():
                for item in self.storage_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    elif item.is_file():
                        item.unlink(missing_ok=True)


session_store = SessionStore()
