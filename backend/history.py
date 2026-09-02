"""
Authoritative persistent analysis history store for SatQuery AI.
Stores all user analyses, executed tool traces, model sources, and evidence.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStore:
    """Lightweight persistent JSON store for SatQuery AI analysis history."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            self.storage_dir = Path(__file__).parent / "data"
        else:
            self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "history.json"
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.history_file.exists():
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_all(self) -> List[Dict[str, Any]]:
        try:
            if not self.history_file.exists():
                return []
            with open(self.history_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception:
            return []

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        temp_file = self.history_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.history_file)

    def create_entry(
        self,
        run_id: str,
        query: str,
        image_names: List[str],
        analysis_type: str = "Unknown",
        status: str = "RUNNING",
    ) -> Dict[str, Any]:
        """Create a new run entry in RUNNING state."""
        now = datetime.now(timezone.utc)
        record = {
            "run_id": run_id,
            "title": self._generate_title(query, analysis_type),
            "query": query,
            "timestamp": now.isoformat(),
            "date_display": now.strftime("%b %d, %Y • %I:%M %p"),
            "image_names": image_names,
            "analysis_type": analysis_type,
            "tools_executed": [],
            "models_used": [],
            "status": status,
            "result_summary": "",
            "full_result": None,
            "error": None,
        }
        with self._lock:
            items = self._read_all()
            # remove any old with same run_id if re-run
            items = [item for item in items if item.get("run_id") != run_id]
            items.insert(0, record)
            self._write_all(items)
        return record

    def update_entry(
        self,
        run_id: str,
        *,
        status: str,
        tools_executed: Optional[List[str]] = None,
        models_used: Optional[List[Dict[str, Any]]] = None,
        result_summary: Optional[str] = None,
        full_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing history entry upon analysis completion or failure."""
        with self._lock:
            items = self._read_all()
            target = None
            for item in items:
                if item.get("run_id") == run_id:
                    target = item
                    break
            if not target:
                return None

            target["status"] = status
            if tools_executed is not None:
                target["tools_executed"] = tools_executed
            if models_used is not None:
                target["models_used"] = models_used
            if result_summary is not None:
                target["result_summary"] = result_summary
            if full_result is not None:
                target["full_result"] = full_result
                if not target.get("result_summary") and full_result.get("answer"):
                    target["result_summary"] = (full_result["answer"][:160] + "...") if len(full_result["answer"]) > 160 else full_result["answer"]
            if error is not None:
                target["error"] = error
            if analysis_type is not None:
                target["analysis_type"] = analysis_type
                target["title"] = self._generate_title(target["query"], analysis_type)

            self._write_all(items)
            return target

    def get_history(self, search: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent history records with optional multi-field search filter."""
        items = self._read_all()
        if search and search.strip():
            term = search.strip().lower()
            filtered = []
            for it in items:
                q = it.get("query", "").lower()
                t = it.get("title", "").lower()
                at = it.get("analysis_type", "").lower()
                imgs = " ".join(it.get("image_names", [])).lower()
                status = it.get("status", "").lower()
                date = it.get("date_display", "").lower()
                models = " ".join([m.get("model", "") + " " + m.get("source", "") for m in it.get("models_used", [])]).lower()
                if (term in q or term in t or term in at or term in imgs or term in status or term in date or term in models):
                    filtered.append(it)
            items = filtered

        summaries = []
        for it in items[:limit]:
            summary = {
                "run_id": it.get("run_id"),
                "title": it.get("title"),
                "query": it.get("query"),
                "timestamp": it.get("timestamp"),
                "date_display": it.get("date_display"),
                "image_names": it.get("image_names", []),
                "analysis_type": it.get("analysis_type"),
                "tools_executed": it.get("tools_executed", []),
                "models_used": it.get("models_used", []),
                "status": it.get("status"),
                "result_summary": it.get("result_summary"),
                "error": it.get("error"),
            }
            summaries.append(summary)
        return summaries

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific run."""
        items = self._read_all()
        for it in items:
            if it.get("run_id") == run_id:
                return it
        return None

    def delete_run(self, run_id: str) -> bool:
        """Delete a single run from history."""
        with self._lock:
            items = self._read_all()
            initial_len = len(items)
            items = [it for it in items if it.get("run_id") != run_id]
            if len(items) < initial_len:
                self._write_all(items)
                return True
            return False

    def clear(self) -> None:
        """Clear all history."""
        with self._lock:
            self._write_all([])

    def _generate_title(self, query: str, analysis_type: str) -> str:
        q_lower = query.lower()
        if "vignan" in q_lower:
            prefix = "Vignan Campus"
        elif "flood" in q_lower or "water" in q_lower:
            prefix = "Hydrological Assessment"
        elif "forest" in q_lower or "vegetation" in q_lower:
            prefix = "Vegetation Assessment"
        elif "urban" in q_lower or "building" in q_lower or "built-up" in q_lower:
            prefix = "Urban Structure Analysis"
        elif "coastal" in q_lower:
            prefix = "Coastal Assessment"
        else:
            prefix = "Satellite Observation"

        if "change" in analysis_type.lower() or "change" in q_lower:
            return f"{prefix} Change Detection"
        elif "fusion" in analysis_type.lower() or "sar" in q_lower:
            return f"{prefix} Optical-SAR Fusion"
        elif "ground" in analysis_type.lower():
            return f"{prefix} Object Grounding"
        elif "caption" in analysis_type.lower():
            return f"{prefix} Scene Description"
        elif "vqa" in analysis_type.lower():
            return f"{prefix} Visual Q&A"
        return f"{prefix} Analysis"


history_store = HistoryStore()
