#!/usr/bin/env python3
"""Isolated public-side JSONL adapter; it has no oracle path or repo lookup."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

# This directory is a copied minimal runtime created by the examiner, not the
# repository root.  `-I` ignores inherited environment/site configuration.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transition_evidence import TransitionEvidenceError, create, load  # noqa: E402


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shelf", required=True, type=Path)
    args = parser.parse_args()
    first = json.loads(sys.stdin.readline())
    if not isinstance(first, dict) or first.get("bootstrap") is not True:
        emit({"ok": False, "code": "BAD_BOOTSTRAP"})
        return 2
    try:
        if set(first) != {"bootstrap", "atlas_b64", "snapshot_b64"}:
            raise ValueError
        atlas_bytes = base64.b64decode(first["atlas_b64"], validate=True)
        snapshot_b64 = first.get("snapshot_b64")
        if snapshot_b64 is None:
            engine = create(atlas_bytes)
        else:
            args.shelf.write_bytes(base64.b64decode(snapshot_b64, validate=True))
            engine = load(args.shelf, atlas_bytes)
        emit({"ok": True, "pid": os.getpid()})
    except (KeyError, ValueError, TypeError, TransitionEvidenceError):
        emit({"ok": False, "code": "BAD_BOOTSTRAP"})
        return 2
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) - {"command", "observation", "action_id", "prediction_id", "step_response"}:
                raise ValueError
            command = request.get("command")
            if command == "bind":
                engine.bind_observation(request["observation"])
                emit({"ok": True})
            elif command == "predict":
                prediction = engine.predict(request["action_id"])
                emit({"ok": True, "prediction": {"id": prediction.id, "action_id": prediction.action_id,
                    "changed": sorted(prediction.changed), "unchanged": sorted(prediction.unchanged), "unknown": sorted(prediction.unknown)}})
            elif command == "accept":
                emit({"ok": True, "experience_id": engine.accept_transition(request["prediction_id"], request["step_response"])})
            elif command == "save":
                engine.save(args.shelf)
                state = engine.snapshot().as_mapping()
                emit({"ok": True, "atlas_id": state["atlas_id"], "version": state["version"]})
            else:
                raise ValueError
        except TransitionEvidenceError as error:
            emit({"ok": False, "code": error.code})
        except (KeyError, TypeError, ValueError):
            emit({"ok": False, "code": "BAD_REQUEST"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
