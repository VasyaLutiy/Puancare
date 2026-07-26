from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from transition_evidence import create, load

from tests.transition_evidence.support import (
    accept,
    atlas_bytes,
    atlas_data,
    changed_observation,
    observation,
    renamed_atlas_and_observation,
    reverse_order,
    snapshot_mapping,
)


SEED = 20260726


def generated_cases() -> list[tuple[int, int, int, int, int, int]]:
    """Fixed seed plus explicit extrema covers every required generation range."""
    cases = [
        (2, 1, 2, 0, 1, 1),
        (3, 2, 3, 1, 2, 2),
        (5, 3, 4, 3, 3, 3),
        (7, 1, 2, 0, 2, 1),
        (9, 4, 4, 5, 1, 3),
    ]
    rng = random.Random(SEED)
    for _ in range(16):
        cases.append((rng.randint(2, 9), rng.randint(1, 4), rng.randint(2, 4), rng.randint(0, 5), rng.randint(1, 3), rng.randint(1, 3)))
    return cases


def generated_atlas(case: tuple[int, int, int, int, int, int]) -> dict:
    variable_count, action_count, domain_size, hypothesis_count, support_min, refute_min = case
    possible = [
        (f"var-{cause}", f"var-{effect}")
        for cause in range(variable_count)
        for effect in range(variable_count)
        if cause != effect
    ]
    hypotheses = [
        {"id": f"source-{index}", "cause": cause, "effect": effect, "relation": "cochanges", "provenance": "atlas"}
        for index, (cause, effect) in enumerate(possible[:hypothesis_count])
    ]
    return atlas_data(
        variable_count=variable_count,
        action_count=action_count,
        domain_size=domain_size,
        hypotheses=hypotheses,
        support_min=support_min,
        refute_min=refute_min,
    )


class MetamorphicTransitionEvidenceTests(unittest.TestCase):
    maxDiff = None

    def test_seeded_fixture_ranges_and_permutation_invariance(self) -> None:
        cases = generated_cases()
        self.assertEqual({2, 3, 4, 5, 6, 7, 8, 9}, {case[0] for case in cases})
        self.assertEqual({1, 2, 3, 4}, {case[1] for case in cases})
        self.assertEqual({2, 3, 4}, {case[2] for case in cases})
        self.assertEqual({0, 1, 2, 3, 4, 5}, {case[3] for case in cases})
        self.assertEqual({1, 2, 3}, {case[4] for case in cases})
        self.assertEqual({1, 2, 3}, {case[5] for case in cases})
        for case in cases:
            with self.subTest(case=case):
                data = generated_atlas(case)
                ordered = create(atlas_bytes(data))
                reordered = create(atlas_bytes(reverse_order(data)))
                self.assertEqual(ordered.snapshot().to_json_bytes(), reordered.snapshot().to_json_bytes())
                self.assertEqual(snapshot_mapping(ordered)["atlas_id"], snapshot_mapping(reordered)["atlas_id"])

    def test_rename_isomorphic_transition_and_save_load_are_equivalent(self) -> None:
        for case in generated_cases():
            with self.subTest(case=case):
                data = generated_atlas(case)
                before = observation(data)
                renamed, renamed_before, variable_map, value_map, action_map = renamed_atlas_and_observation(data, before)
                original = create(atlas_bytes(data))
                isomorphic = create(atlas_bytes(reverse_order(renamed)))
                original.bind_observation(before)
                isomorphic.bind_observation(renamed_before)
                action_id = data["actions"][0]["id"]
                target = data["actions"][0]["intervention"]["variable"]
                changed = {target}
                # A deterministic, non-ID-semantic pattern produces both shared and unshared effects.
                changed.update(item["id"] for index, item in enumerate(data["variables"]) if index % 2 == 1)
                original_after = changed_observation(data, before, changed)
                inverse_variables = {new: old for old, new in variable_map.items()}
                inverse_values = {new: old for old, new in value_map.items()}
                renamed_after = {
                    variable_map[key]: value_map[value]
                    for key, value in original_after.items()
                }
                original_prediction = original.predict(action_id)
                renamed_prediction = isomorphic.predict(action_map[action_id])
                self.assertEqual(
                    set(original_prediction.changed),
                    {inverse_variables[value] for value in renamed_prediction.changed},
                )
                self.assertEqual(
                    set(original_prediction.unchanged),
                    {inverse_variables[value] for value in renamed_prediction.unchanged},
                )
                self.assertEqual(
                    set(original_prediction.unknown),
                    {inverse_variables[value] for value in renamed_prediction.unknown},
                )
                original_id = original.accept_transition(original_prediction.id, {"action_id": action_id, "outcome": "opaque-outcome", "obs_after": original_after})
                renamed_id = isomorphic.accept_transition(renamed_prediction.id, {"action_id": action_map[action_id], "outcome": "opaque-outcome", "obs_after": renamed_after})
                self.assertEqual(original_id, renamed_id)

                def evidence_by_unrenamed_edge(engine, inverse):
                    return sorted(
                        (inverse[item["cause"]], inverse[item["effect"]], item["status"], item["evidence"], item["provenance"])
                        for item in snapshot_mapping(engine)["hypotheses"]
                    )

                identity = {item["id"]: item["id"] for item in data["variables"]}
                self.assertEqual(
                    evidence_by_unrenamed_edge(original, identity),
                    evidence_by_unrenamed_edge(isomorphic, inverse_variables),
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "snapshot.json"
                    isomorphic.save(path)
                    restored = load(path, atlas_bytes(renamed))
                    second = Path(directory) / "snapshot-second.json"
                    restored.save(second)
                    self.assertEqual(path.read_bytes(), second.read_bytes())

