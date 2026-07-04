"""State persistence for discovered channels."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from channel_generator.analyzer import ChannelInfo


class State:
    """JSON-backed state for discovered channels."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        if isinstance(payload, dict):
            self._data = payload

    def save(self) -> None:
        """Persist current state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def merge(self, channels: list[ChannelInfo]) -> None:
        """Merge new channel info into state, preserving first_seen_at.

        Args:
            channels: Newly discovered channels.
        """
        now = datetime.now(UTC).isoformat()
        for ch in channels:
            key = ch.url
            existing = self._data.get(key)
            record = asdict(ch)
            if existing:
                record["first_seen_at"] = existing.get("first_seen_at", record["first_seen_at"])
                record["last_verified_at"] = now
            self._data[key] = record

    def list_channels(
        self,
        max_results: int | None = None,
        min_confidence: str | None = None,
    ) -> list[dict[str, Any]]:
        """List channels sorted by last_verified_at desc, then first_seen_at desc.

        Args:
            max_results: Optional maximum number of results.
            min_confidence: Optional minimum confidence (high > medium > low).

        Returns:
            List of channel records.
        """
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        min_level = confidence_order.get(min_confidence.lower(), 0) if min_confidence else 0
        records = [
            rec
            for rec in self._data.values()
            if confidence_order.get(rec.get("confidence", "low"), 0) >= min_level
        ]
        records.sort(
            key=lambda r: (r.get("last_verified_at", ""), r.get("first_seen_at", "")),
            reverse=True,
        )
        if max_results:
            records = records[:max_results]
        return records

    def get_stats(self) -> dict[str, int]:
        """Return confidence distribution."""
        stats = {"total": len(self._data), "high": 0, "medium": 0, "low": 0}
        for rec in self._data.values():
            conf = rec.get("confidence", "low")
            if conf in stats:
                stats[conf] += 1
        return stats
