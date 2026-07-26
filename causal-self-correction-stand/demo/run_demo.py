"""Recreate the deterministic JSONL demonstration without network access."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transition_evidence import create


def main() -> int:
    atlas_path = ROOT / "examples" / "switch-atlas.yaml"
    journal_path = ROOT / "demo" / "demo.jsonl"
    shelf_path = ROOT / "demo" / "demo-shelf.json"
    engine = create(atlas_path.read_bytes())
    engine.bind_observation({"power": False, "light": False, "alarm": False})
    transitions = (
        ("set_power", "ok", {"power": True, "light": True, "alarm": False}),
        ("trigger_alarm", "ok", {"power": True, "light": True, "alarm": True}),
    )
    lines: list[str] = []
    for action_id, outcome, obs_after in transitions:
        prediction = engine.predict(action_id)
        experience_id = engine.accept_transition(
            prediction.id,
            {"action_id": action_id, "outcome": outcome, "obs_after": obs_after},
        )
        lines.append(json.dumps({
            "event": "accepted_transition",
            "experience_id": experience_id,
            "prediction": {
                "id": prediction.id,
                "action_id": prediction.action_id,
                "changed": sorted(prediction.changed),
                "unchanged": sorted(prediction.unchanged),
                "unknown": sorted(prediction.unknown),
            },
            "outcome": outcome,
            "after": obs_after,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    engine.save(shelf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
