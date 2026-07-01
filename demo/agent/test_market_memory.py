from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.market_memory import MarketMemory


def market_sample(reachable_count: int, score: float) -> dict:
    return {
        "reachable_count": reachable_count,
        "profitable_count": reachable_count,
        "best_net": score,
        "score": score,
    }


def remember(
    memory: MarketMemory,
    driver_id: str,
    minute: int,
    *,
    reachable_count: int,
    score: float,
    lat: float = 23.0,
    lng: float = 113.0,
    k_used: int = 100,
) -> None:
    memory.remember_query(
        driver_id=driver_id,
        current_minute=minute,
        current_lat=lat,
        current_lng=lng,
        raw_items=[{} for _ in range(20)],
        reachable_items=[{} for _ in range(reachable_count)],
        market_sample=market_sample(reachable_count, score),
        k_used=k_used,
        query_scan_minutes=max(0, k_used // 10),
    )


class MarketMemoryTest(unittest.TestCase):
    def test_cold_start_query_uses_explore_k(self) -> None:
        memory = MarketMemory()

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=8 * 60,
            current_lat=23.0,
            current_lng=113.0,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 600)
        self.assertEqual(plan["reason"], "driver_cold_start")

    def test_query_k_returns_normal_after_explore_budget(self) -> None:
        memory = MarketMemory()
        for day in range(5):
            for seq in range(2):
                remember(
                    memory,
                    "D001",
                    day * 1440 + 8 * 60 + seq,
                    reachable_count=2,
                    score=300.0,
                    lat=23.0 + day * 0.1,
                    lng=113.0,
                    k_used=600,
                )

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=6 * 1440 + 8 * 60,
            current_lat=24.5,
            current_lng=113.0,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 100)
        self.assertEqual(plan["reason"], "explore_budget_or_cooldown")

    def test_explore_budget_uses_full_history_not_recent_window_only(self) -> None:
        memory = MarketMemory()
        for day in range(5):
            for seq in range(2):
                remember(
                    memory,
                    "D001",
                    day * 1440 + 8 * 60 + seq,
                    reachable_count=2,
                    score=300.0,
                    lat=23.0 + day * 0.1,
                    lng=113.0,
                    k_used=600,
                )

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=20 * 1440 + 8 * 60,
            current_lat=25.5,
            current_lng=113.0,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 100)
        self.assertEqual(plan["reason"], "explore_budget_or_cooldown")

    def test_query_k_respects_preference_time_cap(self) -> None:
        memory = MarketMemory()

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=8 * 60,
            current_lat=23.0,
            current_lng=113.0,
            max_query_minutes=10,
        )

        self.assertEqual(plan["k"], 100)
        self.assertEqual(plan["reason"], "preference_time_cap")

    def test_failed_explore_query_cools_down_same_bucket(self) -> None:
        memory = MarketMemory()
        remember(memory, "D001", 8 * 60, reachable_count=0, score=-10000.0, k_used=600)

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=8 * 60 + 30,
            current_lat=23.0,
            current_lng=113.0,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 100)
        self.assertEqual(plan["reason"], "explore_budget_or_cooldown")

    def test_truncated_normal_query_can_escalate_to_explore(self) -> None:
        memory = MarketMemory()
        for day in range(3):
            remember(memory, "D001", day * 1440 + 7 * 60, reachable_count=2, score=300.0, k_used=600)
        memory.remember_query(
            driver_id="D001",
            current_minute=4 * 1440 + 8 * 60,
            current_lat=23.0,
            current_lng=113.0,
            raw_items=[{} for _ in range(100)],
            reachable_items=[],
            market_sample=market_sample(0, -10000.0),
            k_used=100,
            query_scan_minutes=10,
        )

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=4 * 1440 + 8 * 60 + 15,
            current_lat=23.0,
            current_lng=113.0,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 600)
        self.assertEqual(plan["reason"], "normal_query_truncated_no_reachable")

    def test_new_grid_hour_uses_normal_probe_after_bootstrap(self) -> None:
        memory = MarketMemory()
        for day in range(3):
            remember(memory, "D001", day * 1440 + 7 * 60, reachable_count=2, score=300.0, k_used=600)

        plan = memory.choose_query_k(
            driver_id="D001",
            current_minute=4 * 1440 + 8 * 60,
            current_lat=23.8,
            current_lng=113.8,
            max_query_minutes=60,
        )

        self.assertEqual(plan["k"], 100)
        self.assertEqual(plan["reason"], "new_grid_hour_probe")

    def test_cold_start_keeps_wait_one(self) -> None:
        memory = MarketMemory()
        for offset in range(3):
            remember(memory, "D001", 8 * 60 + offset, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D001",
            current_minute=8 * 60 + 3,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 15)
        self.assertEqual(plan["reason"], "cold_start")

    def test_poor_hour_waits_until_nearby_good_hour(self) -> None:
        memory = MarketMemory()
        previous_day = 0
        current_day = 1440
        for offset in range(4):
            remember(memory, "D001", previous_day + 8 * 60 + offset, reachable_count=0, score=-10000.0)
        for offset in range(4):
            remember(memory, "D001", previous_day + 9 * 60 + offset, reachable_count=3, score=300.0)
        remember(memory, "D001", current_day + 8 * 60 + 45, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D001",
            current_minute=current_day + 8 * 60 + 45,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 15)
        self.assertEqual(plan["reason"], "wait_for_historically_good_hour")

    def test_driver_histories_are_isolated(self) -> None:
        memory = MarketMemory()
        for offset in range(8):
            remember(memory, "D001", 8 * 60 + offset, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D002",
            current_minute=8 * 60 + 10,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 15)
        self.assertEqual(plan["reason"], "cold_start")

    def test_good_hour_from_other_location_is_not_used(self) -> None:
        memory = MarketMemory()
        previous_day = 0
        current_day = 1440
        for offset in range(4):
            remember(memory, "D001", previous_day + 8 * 60 + offset, reachable_count=0, score=-10000.0)
        for offset in range(4):
            remember(
                memory,
                "D001",
                previous_day + 9 * 60 + offset,
                reachable_count=3,
                score=300.0,
                lat=24.0,
                lng=114.0,
            )
        remember(memory, "D001", current_day + 8 * 60 + 45, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D001",
            current_minute=current_day + 8 * 60 + 45,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 15)
        self.assertEqual(plan["reason"], "no_recent_better_market")

    def test_wait_is_capped_when_good_hour_is_farther_away(self) -> None:
        memory = MarketMemory()
        previous_day = 0
        current_day = 1440
        for offset in range(4):
            remember(memory, "D001", previous_day + 8 * 60 + offset, reachable_count=0, score=-10000.0)
        for offset in range(4):
            remember(memory, "D001", previous_day + 12 * 60 + offset, reachable_count=3, score=300.0)
        remember(memory, "D001", current_day + 8 * 60 + 45, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D001",
            current_minute=current_day + 8 * 60 + 45,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 30)
        self.assertEqual(plan["reason"], "wait_for_historically_good_hour")

    def test_repeated_no_reachable_without_better_hour_keeps_wait_one(self) -> None:
        memory = MarketMemory()
        for offset in range(8):
            remember(memory, "D001", 10 * 60 + offset, reachable_count=0, score=-10000.0)

        plan = memory.suggest_no_reachable_wait(
            driver_id="D001",
            current_minute=10 * 60 + 8,
            current_lat=23.0,
            current_lng=113.0,
            market_sample=market_sample(0, -10000.0),
        )

        self.assertEqual(plan["duration_minutes"], 15)
        self.assertEqual(plan["reason"], "no_recent_better_market")

    def test_market_state_exposes_grid_hour_profile(self) -> None:
        memory = MarketMemory()
        for offset in range(8):
            remember(memory, "D001", 10 * 60 + offset, reachable_count=2, score=300.0)

        state = memory.market_state(
            driver_id="D001",
            current_minute=10 * 60 + 10,
            current_lat=23.0,
            current_lng=113.0,
        )

        self.assertEqual(state["state"], "good")
        self.assertEqual(state["query_count"], 8)
        self.assertEqual(state["avg_reachable_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
