"""Group contract-compatible transition traces without concatenating arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from arena.imitation.contract import TraceSequence


@dataclass(frozen=True, slots=True)
class TraceCorpus:
    """A corpus that refuses schema or selected provenance drift."""

    traces: tuple[TraceSequence, ...]
    compatibility_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        traces = tuple(self.traces)
        keys = tuple(sorted(self.compatibility_keys))
        if not traces or not all(isinstance(item, TraceSequence) for item in traces):
            raise ValueError("a trace corpus requires at least one TraceSequence")
        contract = traces[0].contract.sha256
        if any(item.contract.sha256 != contract for item in traces[1:]):
            raise ValueError("trace corpus contracts differ")
        for key in keys:
            if any(key not in item.parameters for item in traces):
                raise ValueError(f"trace corpus is missing compatibility key: {key}")
            expected = traces[0].parameters[key]
            if any(item.parameters[key] != expected for item in traces[1:]):
                raise ValueError(f"trace corpus provenance differs: {key}")
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "compatibility_keys", keys)

    @property
    def rows(self) -> int:
        return sum(item.rows for item in self.traces)

    @property
    def contract_sha256(self) -> str:
        return self.traces[0].contract.sha256

    @property
    def compatibility(self):
        first = self.traces[0].parameters
        return MappingProxyType({key: first[key] for key in self.compatibility_keys})

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "arena_trace_corpus_v1",
            "contract_sha256": self.contract_sha256,
            "trace_count": len(self.traces),
            "rows": self.rows,
            "compatibility": dict(self.compatibility),
            "trace_identity_sha256": tuple(
                item.identity_sha256 for item in self.traces
            ),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest().upper()

    def append(self, *traces: TraceSequence) -> TraceCorpus:
        return TraceCorpus(self.traces + tuple(traces), self.compatibility_keys)


__all__ = ["TraceCorpus"]
