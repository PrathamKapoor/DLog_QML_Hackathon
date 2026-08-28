"""
Stored-artifact loading with explicit provenance.

The notebook must be able to show results instantly to a judge without waiting
four minutes for a 20-seed sweep. That is only acceptable if a cached result is
**visibly** labelled as cached. This module makes provenance a first-class,
non-optional part of every load:

    payload, provenance = run_or_load(...)
    print(provenance.banner())     # -> "CACHED  multi_seed  (written 2026-08-28 ...)"

There is deliberately no code path that returns data without a provenance
object, and no way to make a cached load *look* fresh.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class Provenance:
    """Where a result came from. Always displayed alongside the numbers."""

    source: str                 # "fresh" | "cached"
    name: str
    path: Optional[str] = None
    written_at: Optional[str] = None
    runtime_seconds: Optional[float] = None

    @property
    def is_fresh(self) -> bool:
        return self.source == "fresh"

    def banner(self) -> str:
        if self.is_fresh:
            timing = (f" in {self.runtime_seconds:.1f}s"
                      if self.runtime_seconds is not None else "")
            return f"[FRESHLY COMPUTED]  {self.name}{timing}"
        stamp = f", written {self.written_at}" if self.written_at else ""
        return f"[CACHED ARTIFACT]  {self.name}  (loaded from disk{stamp})"

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "name": self.name, "path": self.path,
                "written_at": self.written_at,
                "runtime_seconds": self.runtime_seconds}


def _timestamp(path: str) -> Optional[str]:
    try:
        return _dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%Y-%m-%d %H:%M:%S")
    except OSError:                                        # pragma: no cover
        return None


def load_artifact(path: str, name: Optional[str] = None
                  ) -> Tuple[Optional[Any], Provenance]:
    """Load a stored JSON artifact. Returns (None, provenance) when absent."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    if not os.path.exists(path):
        return None, Provenance(source="missing", name=name, path=path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, Provenance(source="cached", name=name, path=os.path.abspath(path),
                               written_at=_timestamp(path))


def run_or_load(name: str, path: str, compute: Callable[[], Any],
                prefer_cached: bool = True) -> Tuple[Any, Provenance]:
    """
    Return a stored artifact when available and `prefer_cached`, else compute it.

    `prefer_cached=False` always recomputes, which is what the notebook's
    `RUN_FRESH_EXPERIMENT` switch sets. Either way the returned `Provenance`
    states unambiguously which happened.
    """
    if prefer_cached:
        payload, provenance = load_artifact(path, name=name)
        if payload is not None:
            return payload, provenance

    import time
    start = time.perf_counter()
    payload = compute()
    elapsed = time.perf_counter() - start
    return payload, Provenance(source="fresh", name=name,
                               runtime_seconds=round(elapsed, 2))


def require(payload: Optional[Any], provenance: Provenance, hint: str) -> Any:
    """Fail loudly, with an actionable message, rather than silently degrading."""
    if payload is None:
        raise FileNotFoundError(
            f"Artifact '{provenance.name}' not found at {provenance.path}.\n"
            f"  {hint}"
        )
    return payload
