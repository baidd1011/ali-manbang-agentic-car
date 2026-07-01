from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.hotspot_planner import HotspotPlanner


def cargo_item(
    cargo_id: str,
    *,
    start_lat: float = 23.6,
    start_lng: float = 113.6,
    end_lat: float = 23.7,
    end_lng: float = 113.7,
    price: int = 200000,
    distance_km: float = 3.0,
) -> dict:
    return {
        "distance_km": distance_km,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": "普通货物",
            "price": price,
            "start": {"lat": start_lat, "lng": start_lng},
            "end": {"lat": end_lat, "lng": end_lng},
        },
    }


def hotspot_items(count: int = 6, **kwargs) -> list[dict]:
    return [cargo_item(f"C{i}", **kwargs) for i in range(count)]


class HotspotPlannerTest(unittest.TestCase):
    def test_cold_start_does_not_reposition(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(2)
        planner.observe("D001", 9 * 60, items, [])

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNone(plan)

    def test_no_reachable_uses_learned_hotspot(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["action"]["action"], "reposition")
        self.assertAlmostEqual(plan["action"]["params"]["latitude"], 23.6)
        self.assertAlmostEqual(plan["action"]["params"]["longitude"], 113.6)
        self.assertEqual(plan["meta"]["action_reason"], "no_reachable")

    def test_hotspot_too_far_is_ignored(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6, start_lat=26.0, start_lng=116.0, end_lat=26.1, end_lng=116.1)
        planner.observe("D001", 9 * 60, items, items)

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNone(plan)

    def test_low_market_can_trigger_reposition(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)
        for score in [300.0, 320.0, 350.0, 400.0, 450.0, 500.0]:
            planner.remember_market_sample("D001", {"score": score})

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": 250.0}, True)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["meta"]["action_reason"], "low_market")

    def test_good_market_does_not_reposition(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)
        for score in [300.0, 320.0, 350.0, 400.0, 450.0, 500.0]:
            planner.remember_market_sample("D001", {"score": score})

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": 900.0}, True)

        self.assertIsNone(plan)

    def test_driver_samples_are_isolated(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)

        plan = planner.plan_reposition("D002", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNone(plan)

    def test_near_hotspot_does_not_reposition(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)

        plan = planner.plan_reposition("D001", 9 * 60, 23.601, 113.601, {"score": -10000.0}, False)

        self.assertIsNone(plan)

    def test_failed_hotspot_is_debounced(self) -> None:
        planner = HotspotPlanner()
        items = hotspot_items(6)
        planner.observe("D001", 9 * 60, items, items)
        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)
        self.assertIsNotNone(plan)
        planner.remember_reposition("D001", plan, 9 * 60)

        next_plan = planner.plan_reposition("D001", 10 * 60, 23.6, 113.6, {"score": -10000.0}, False)

        self.assertIsNone(next_plan)

    def test_observed_only_raw_items_need_multiple_observations(self) -> None:
        planner = HotspotPlanner()
        raw_items = hotspot_items(6, price=900000)
        planner.observe("D001", 9 * 60, raw_items, [])

        plan = planner.plan_reposition("D001", 9 * 60, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNone(plan)

    def test_observed_only_raw_items_can_seed_no_reachable_hotspot_after_learning(self) -> None:
        planner = HotspotPlanner()
        raw_items = hotspot_items(6, price=900000)
        for minute in [9 * 60, 9 * 60 + 30, 9 * 60 + 60]:
            planner.observe("D001", minute, raw_items, [])

        plan = planner.plan_reposition("D001", 9 * 60 + 90, 23.0, 113.0, {"score": -10000.0}, False)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["action"]["action"], "reposition")
        self.assertEqual(plan["meta"]["reachable_sample_count"], 0)
        self.assertAlmostEqual(plan["meta"]["confidence"], 0.55)


if __name__ == "__main__":
    unittest.main()
