from __future__ import annotations

import unittest

from transition_evidence import create

from tests.transition_evidence.support import (
    accept,
    atlas_bytes,
    atlas_data,
    changed_observation,
    hypothesis_by_edge,
    observation,
    snapshot_mapping,
)


class ActionTargetEvidenceScopeTests(unittest.TestCase):
    def test_informative_action_updates_only_hypotheses_caused_by_its_target(self) -> None:
        data = atlas_data(
            variable_count=4,
            hypotheses=[
                {"id": "incidental-source", "cause": "var-2", "effect": "var-3", "relation": "cochanges", "provenance": "atlas"}
            ],
        )
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        # Target A and incidental C both move, while D does not.
        after = changed_observation(data, before, {"var-0", "var-2"})
        accept(engine, "action-0", after)

        target_effect = hypothesis_by_edge(engine, "var-0", "var-1")
        incidental = hypothesis_by_edge(engine, "var-2", "var-3")
        self.assertEqual({"supports": 0, "contradicts": 1}, target_effect["evidence"])
        self.assertEqual({"supports": 0, "contradicts": 0}, incidental["evidence"])
        self.assertEqual((), incidental["supporting_evidence"])
        self.assertEqual((), incidental["contradicting_evidence"])

    def test_uninformative_target_still_records_after_but_changes_no_candidates_or_evidence(self) -> None:
        data = atlas_data(
            variable_count=4,
            hypotheses=[
                {"id": "incidental-source", "cause": "var-2", "effect": "var-3", "relation": "cochanges", "provenance": "atlas"}
            ],
        )
        engine = create(atlas_bytes(data))
        before = observation(data)
        engine.bind_observation(before)
        # The selected target A is unchanged, even though C and D change.
        after = changed_observation(data, before, {"var-2", "var-3"})
        accept(engine, "action-0", after, outcome="target-declined")

        state = snapshot_mapping(engine)
        self.assertEqual(1, state["version"])
        self.assertEqual(after, dict(state["current_observation"]))
        self.assertEqual(1, len(state["experiences"]))
        self.assertEqual(1, len(state["hypotheses"]), "no target candidates are born")
        incidental = hypothesis_by_edge(engine, "var-2", "var-3")
        self.assertEqual({"supports": 0, "contradicts": 0}, incidental["evidence"])

