"""Canonical JSON helpers used for identity, comparison and snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .errors import AtlasValidationError
from .types import JsonScalar


def is_json_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return isinstance(value, float) and math.isfinite(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def scalar_token(value: JsonScalar) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def content_id(cause: str, effect: str, relation: str) -> str:
    return hashlib.sha256(canonical_json_bytes([cause, effect, relation])).hexdigest()


def sha256_commitment(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atlas_digest(canonical_atlas: object) -> str:
    return sha256_commitment(canonical_atlas)


def genesis_chain_commitment(atlas_id: str) -> str:
    return sha256_commitment({"atlas_id": atlas_id, "kind": "transition-shelf/genesis/v1"})


def genesis_entry_commitment(atlas_id: str) -> str:
    """Deterministic predecessor for the first committed experience."""
    return sha256_commitment({"atlas_id": atlas_id, "kind": "transition-shelf/entry-genesis/v1"})


def post_chain_commitment(pre_chain_commitment: str, entry_commitment: str) -> str:
    return sha256_commitment([pre_chain_commitment, entry_commitment])


def pointer_join(path: str, token: str | int) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def ensure_json_scalar(value: object, path: str) -> JsonScalar:
    if not is_json_scalar(value):
        raise AtlasValidationError("BAD_DOMAIN", path, "expected a finite JSON scalar")
    return value  # type: ignore[return-value]
