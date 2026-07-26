"""Stable public errors for the transition evidence engine."""

from __future__ import annotations


class TransitionEvidenceError(Exception):
    """A machine-readable engine error.

    ``code`` is stable API and ``path`` is an RFC 6901 JSON Pointer.  The
    exception text is diagnostic only and intentionally not part of the API.
    """

    def __init__(self, code: str, path: str = "", message: str | None = None) -> None:
        self.code = code
        self.path = path
        super().__init__(message or f"{code} at {path or '/'}")


class AtlasValidationError(TransitionEvidenceError):
    """The atlas document is unsafe or does not conform to the v1 schema."""


class RuntimeValidationError(TransitionEvidenceError):
    """An observation, prediction, or environment response is invalid."""


class SnapshotValidationError(TransitionEvidenceError):
    """A persisted shelf is invalid or belongs to another atlas."""


class PersistenceError(TransitionEvidenceError):
    """An atomic persistence operation could not be completed."""
