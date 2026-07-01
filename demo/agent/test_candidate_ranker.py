from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.candidate_ranker import CandidateRanker


def item(cargo_id: str, *, price: float, distance_km: float, end_lat: float = 23.1) -> dict[str, Any]:
    return {
        "distance_km": distance_km,
        "pickup_minutes": int(distance_km),
        "wait_before_loading_minutes": 0,
        "finish_minutes": 120,
        "minutes_until_load_deadline": 90,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": "普通货物",
            "price": price,
            "cost_time_minutes": 60,
            "start": {"lat": 23.0, "lng": 113.0},
            "end": {"lat": end_lat, "lng": 113.2},
        },
    }


class CandidateRankerTest(unittest.TestCase):
    def test_rank_items_prefers_high_net_even_if_not_nearest(self) -> None:
        ranker = CandidateRanker()
        items = [item(f"near_{i}", price=120.0, distance_km=1.0) for i in range(25)]
        items.append(item("valuable_far", price=1600.0, distance_km=30.0))

        ranked = ranker.rank_items(items, current_minute=30)

        self.assertEqual(ranked[0]["cargo"]["cargo_id"], "valuable_far")

    def test_deterministic_take_requires_clear_gap(self) -> None:
        ranker = CandidateRanker()
        ranked = ranker.rank_items(
            [
                item("best", price=1800.0, distance_km=5.0),
                item("second", price=300.0, distance_km=5.0),
            ],
            current_minute=30,
        )

        action = ranker.deterministic_take_action(ranked)

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "best"}})

    def test_close_candidates_fall_back_to_model(self) -> None:
        ranker = CandidateRanker()
        ranked = ranker.rank_items(
            [
                item("best", price=500.0, distance_km=5.0),
                item("second", price=470.0, distance_km=5.0),
            ],
            current_minute=30,
        )

        self.assertIsNone(ranker.deterministic_take_action(ranked))


if __name__ == "__main__":
    unittest.main()
