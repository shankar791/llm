"""
Session Context Builder for SatQuery AI (Step 2).
Converts in-memory session data into clean, structured, LLM-ready context.
Enforces strict anti-hallucination rules (zero fact fabrication) and compact token budgets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .session import session_store, SessionStore
except ImportError:
    from session import session_store, SessionStore


@dataclass
class SessionContextResult:
    """Deterministic structured context for subsequent LLM prompt construction."""
    session_id: str
    query: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    session_info: Dict[str, Any] = field(default_factory=dict)
    text_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "messages": self.messages,
            "images": self.images,
            "session_info": self.session_info,
            "text_context": self.text_context,
        }


class SessionContextBuilder:
    """
    Constructs compact, grounded conversation context from session memory.
    Strictly forbids data invention, fake coordinates, or metric fabrication.
    """

    def __init__(
        self,
        session_store_instance: Optional[SessionStore] = None,
        default_max_messages: int = 6,
        default_max_images: int = 5,
    ):
        self.session_store = session_store_instance or session_store
        self.default_max_messages = default_max_messages
        self.default_max_images = default_max_images

    def build_context(
        self,
        session_id: str,
        query: str,
        max_recent_messages: Optional[int] = None,
        max_images: Optional[int] = None,
    ) -> SessionContextResult:
        """
        Build a structured, LLM-ready context for session_id and current user query.
        """
        max_msgs = max_recent_messages if max_recent_messages is not None else self.default_max_messages
        max_imgs = max_images if max_images is not None else self.default_max_images

        # 1. Retrieve session memory
        sess = self.session_store.get_session_memory(session_id)
        if sess is None:
            # Fallback to get_session for backward compatibility with persisted sessions
            sess = self.session_store.get_session(session_id)

        clean_query = query.strip() if query else ""

        # If session does not exist or is empty
        if not sess:
            empty_info = {
                "active_image_ids": [],
                "relevant_analysis_results": [],
                "session_metadata": {},
            }
            text_context = self._format_text_context(clean_query, [], [], empty_info)
            return SessionContextResult(
                session_id=session_id,
                query=clean_query,
                messages=[],
                images=[],
                session_info=empty_info,
                text_context=text_context,
            )

        # 2. Select recent conversation messages (preserving chronological order)
        raw_conversation = sess.get("conversation") or sess.get("messages") or []
        selected_messages = self._select_recent_messages(raw_conversation, max_msgs, current_query=clean_query)

        # 3. Select relevant images and sanitize (exclude raw binary data)
        raw_images = sess.get("images") or []
        # If images list is empty but single image exists (from legacy sessions)
        if not raw_images and sess.get("image"):
            legacy_img = sess.get("image", {})
            raw_images = [{
                "image_id": legacy_img.get("image_id", "img_primary"),
                "filename": (legacy_img.get("filenames") or ["image.png"])[0],
                "image_path": (legacy_img.get("raster_paths") or [None])[0],
                "analysis": sess.get("analysis"),
                "task": (sess.get("tool_results") or [{}])[0].get("tool", "analysis"),
                "evidence": sess.get("evidence", []),
                "gis_results": {},
            }]

        selected_images = self._select_and_sanitize_images(raw_images, max_imgs)

        # 4. Extract session information
        raw_context = sess.get("context") or {}
        active_ids = list(raw_context.get("active_image_ids", []))
        if not active_ids and selected_images:
            active_ids = [img["image_id"] for img in selected_images]

        relevant_results = raw_context.get("relevant_analysis_results", [])
        if not relevant_results and sess.get("analysis"):
            relevant_results = [sess.get("analysis")]

        session_info = {
            "active_image_ids": active_ids,
            "relevant_analysis_results": relevant_results,
            "session_metadata": raw_context.get("session_metadata") or sess.get("metadata") or {},
        }

        # 5. Format deterministic text context
        text_context = self._format_text_context(clean_query, selected_messages, selected_images, session_info)

        return SessionContextResult(
            session_id=session_id,
            query=clean_query,
            messages=selected_messages,
            images=selected_images,
            session_info=session_info,
            text_context=text_context,
        )

    def _select_recent_messages(
        self,
        conversation: List[Dict[str, Any]],
        max_messages: int,
        current_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select up to max_messages recent turns, preserving chronological order.
        If current_query is provided and matches the latest user message, excludes it
        from prior history so it appears only in the active query section.
        Sanitizes message dicts to role, content, timestamp.
        """
        if not conversation:
            return []

        history = conversation
        if current_query and history:
            last_msg = history[-1]
            if last_msg.get("role") == "user" and last_msg.get("content", "").strip() == current_query.strip():
                history = history[:-1]

        sliced = history[-max_messages:] if max_messages > 0 else []
        clean_messages = []
        for msg in sliced:
            clean_messages.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", "")),
                "timestamp": str(msg.get("timestamp", "")),
            })
        return clean_messages

    def _select_and_sanitize_images(
        self,
        images: List[Dict[str, Any]],
        max_images: int,
    ) -> List[Dict[str, Any]]:
        """
        Select up to max_images records and sanitize to prevent raw byte leaks.
        Preserves individual image IDs, filenames, analysis, task, evidence, and GIS metrics exactly.
        """
        if not images:
            return []

        # Take up to max_images (most recent images)
        selected = images[-max_images:] if max_images > 0 else []
        clean_images = []

        for img in selected:
            # Strip any raw bytes, base64 data, or tensor data
            clean_entry = {
                "image_id": str(img.get("image_id", "")),
                "filename": str(img.get("filename", "")),
                "image_path": str(img.get("image_path") or img.get("image_ref") or ""),
                "task": str(img.get("task", "analysis")),
                "analysis": copy.deepcopy(img.get("analysis")),
                "evidence": copy.deepcopy(img.get("evidence") or []),
                "gis_results": copy.deepcopy(img.get("gis_results") or {}),
            }

            # If metadata exists and has no raw bytes, include safe keys
            if img.get("metadata") and isinstance(img["metadata"], dict):
                safe_meta = {
                    k: v for k, v in img["metadata"].items()
                    if not isinstance(v, (bytes, bytearray)) and "b64" not in k.lower()
                }
                clean_entry["metadata"] = safe_meta

            clean_images.append(clean_entry)

        return clean_images

    def _format_text_context(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        session_info: Dict[str, Any],
    ) -> str:
        """
        Format clean, deterministic Markdown text context for later LLM prompting.
        Strict anti-hallucination: only includes factual items stored in memory.
        """
        lines = []

        # 1. Current Query
        lines.append("### CURRENT USER QUERY:")
        lines.append(query if query else "(No query provided)")
        lines.append("")

        # 2. Recent Conversation History
        lines.append("### RECENT CONVERSATION HISTORY:")
        if messages:
            for msg in messages:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "").strip()
                lines.append(f"- {role_label}: {content}")
        else:
            lines.append("No prior conversation in this session.")
        lines.append("")

        # 3. Analyzed Images & Verified Evidence
        lines.append("### ANALYZED IMAGES & VERIFIED FINDINGS:")
        if images:
            for img in images:
                iid = img.get("image_id", "Unknown")
                fn = img.get("filename", "")
                task = img.get("task", "analysis")
                lines.append(f"Image [{iid}] ({fn}):")
                lines.append(f"  * Tool/Task: {task}")

                # Analysis summary
                analysis = img.get("analysis")
                if isinstance(analysis, dict):
                    ans_text = analysis.get("answer") or analysis.get("finding") or str(analysis)
                    lines.append(f"  * Previous Analysis: {ans_text}")
                elif analysis:
                    lines.append(f"  * Previous Analysis: {analysis}")

                # Verified evidence items
                evidence = img.get("evidence", [])
                if evidence:
                    ev_strs = []
                    for ev in evidence:
                        if isinstance(ev, dict):
                            eid = ev.get("evidence_id", "E")
                            lbl = ev.get("label", "detection")
                            pct = ev.get("coverage_pct")
                            if pct is not None:
                                ev_strs.append(f"{eid}: {lbl} (~{pct}%)")
                            else:
                                ev_strs.append(f"{eid}: {lbl}")
                        else:
                            ev_strs.append(str(ev))
                    lines.append(f"  * Verified Evidence: {', '.join(ev_strs)}")

                # GIS Results
                gis = img.get("gis_results", {})
                if gis and isinstance(gis, dict):
                    gis_strs = [f"{k}={v}" for k, v in sorted(gis.items()) if not isinstance(v, (bytes, bytearray))]
                    if gis_strs:
                        lines.append(f"  * GIS Metrics: {', '.join(gis_strs)}")
        else:
            lines.append("No images analyzed in this session yet.")
        lines.append("")

        # 4. Session Active Image IDs
        active_ids = session_info.get("active_image_ids", [])
        lines.append("### SESSION OVERVIEW:")
        if active_ids:
            lines.append(f"Active Image IDs: {', '.join(active_ids)}")
        else:
            lines.append("Active Image IDs: None")

        return "\n".join(lines)


# Module-level convenience function
def build_session_context(
    session_id: str,
    query: str,
    max_recent_messages: int = 6,
    max_images: int = 5,
    session_store_instance: Optional[SessionStore] = None,
) -> Dict[str, Any]:
    """
    Build structured, LLM-ready context from session memory and current user query.
    Returns a dictionary suitable for JSON serialization and prompt building.
    """
    builder = SessionContextBuilder(
        session_store_instance=session_store_instance,
        default_max_messages=max_recent_messages,
        default_max_images=max_images,
    )
    res = builder.build_context(
        session_id=session_id,
        query=query,
        max_recent_messages=max_recent_messages,
        max_images=max_images,
    )
    return res.to_dict()
