"""Import shim (2026-10-18): the session's scribe lives in ``shared_state/writers.py``; this path stays importable."""

from experimental.neural_agent.shared_state.writers import SessionScribe, information_request_record

__all__ = ["SessionScribe", "information_request_record"]
