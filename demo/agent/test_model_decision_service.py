from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.model_decision_service import ModelDecisionService
from agent.market_memory import MarketMemory


class FakeApi:
    def __init__(self) -> None:
        self.status = {
            "simulation_progress_minutes": 30,
            "current_lat": 23.0,
            "current_lng": 113.0,
            "truck_length": 4.2,
            "completed_order_count": 0,
            "preferences": [],
        }
        self.items: list[dict[str, Any]] = []
        self.query_cargo_calls = 0
        self.query_cargo_k_values: list[int] = []
        self.model_calls = 0

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        return dict(self.status)

    def query_cargo(self, driver_id: str, latitude: float, longitude: float, k: int = 100) -> dict[str, Any]:
        self.query_cargo_calls += 1
        self.query_cargo_k_values.append(k)
        return {"items": self.items}

    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]:
        return {"records": []}

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.model_calls += 1
        raise AssertionError("model should not be called")


class FakeCompiler:
    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = rules

    def compile(self, driver_id: str, preferences: Any) -> list[dict[str, Any]]:
        return self.rules


def item(
    cargo_id: str,
    *,
    remove_time: str,
    load_time: Any = None,
    price: float = 0.0,
    distance_km: float = 0.0,
) -> dict[str, Any]:
    cargo: dict[str, Any] = {
        "cargo_id": cargo_id,
        "remove_time": remove_time,
        "start": {"lat": 23.0, "lng": 113.0},
        "end": {"lat": 23.1, "lng": 113.1},
        "price": price,
        "cost_time_minutes": 60,
    }
    if load_time is not None:
        cargo["load_time"] = load_time
    return {"distance_km": distance_km, "cargo": cargo}


class ModelDecisionServiceCargoFilterTest(unittest.TestCase):
    def test_filters_expired_and_invalid_load_time_before_model_choice(self) -> None:
        service = ModelDecisionService(FakeApi())  # type: ignore[arg-type]
        decision_minute = 60
        items = [
            item("expired", remove_time="2026-03-01 01:00:00", load_time=["2026-03-01 01:00:00", "2026-03-01 02:00:00"]),
            item("invalid_window", remove_time="2026-03-01 03:00:00", load_time=["2026-03-01 03:00:00", "2026-03-01 02:00:00"]),
            item("valid_window", remove_time="2026-03-01 03:00:00", load_time=["2026-03-01 01:00:00", "2026-03-01 02:00:00"]),
            item("no_window", remove_time="2026-03-01 03:00:00"),
        ]

        filtered = service._filter_reachable_cargo_items(items, decision_minute)

        cargo_ids = [str(row["cargo"]["cargo_id"]) for row in filtered]
        self.assertEqual(cargo_ids, ["valid_window", "no_window"])

    def test_pre_query_must_take_waits_when_cargo_not_reachable(self) -> None:
        api = FakeApi()
        service = ModelDecisionService(api)  # type: ignore[arg-type]
        service._preference_compiler = FakeCompiler(  # type: ignore[assignment]
            [
                {
                    "rule_type": "must_take_cargo",
                    "params": {
                        "cargo_id": "MUST",
                        "pickup_lat": 23.0,
                        "pickup_lng": 113.0,
                        "active_start_minute": 0,
                        "active_end_minute": 120,
                    },
                }
            ]
        )

        action = service.decide("D001")

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 15}})
        self.assertEqual(api.query_cargo_calls, 1)
        self.assertEqual(api.query_cargo_k_values, [600])
        self.assertEqual(api.model_calls, 0)

    def test_no_reachable_branch_uses_dynamic_market_wait(self) -> None:
        api = FakeApi()
        current_minute = 1440 + 8 * 60 + 45
        api.status["simulation_progress_minutes"] = current_minute
        service = ModelDecisionService(api)  # type: ignore[arg-type]
        memory = MarketMemory()
        for offset in range(4):
            memory.remember_query(
                driver_id="D001",
                current_minute=8 * 60 + offset,
                current_lat=23.0,
                current_lng=113.0,
                raw_items=[],
                reachable_items=[],
                market_sample={"reachable_count": 0, "profitable_count": 0, "best_net": 0.0, "score": -10000.0},
            )
        for offset in range(4):
            memory.remember_query(
                driver_id="D001",
                current_minute=9 * 60 + offset,
                current_lat=23.0,
                current_lng=113.0,
                raw_items=[{}],
                reachable_items=[{}],
                market_sample={"reachable_count": 1, "profitable_count": 1, "best_net": 300.0, "score": 300.0},
            )
        service._market_memory = memory  # type: ignore[assignment]

        action = service.decide("D001")

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 15}})
        self.assertEqual(api.model_calls, 0)

    def test_dynamic_market_wait_is_capped_before_must_take_prepare_window(self) -> None:
        api = FakeApi()
        current_minute = 1440 + 8 * 60 + 45
        api.status["simulation_progress_minutes"] = current_minute
        service = ModelDecisionService(api)  # type: ignore[arg-type]
        service._preference_compiler = FakeCompiler(  # type: ignore[assignment]
            [
                {
                    "rule_type": "must_take_cargo",
                    "params": {
                        "cargo_id": "MUST",
                        "pickup_lat": 23.0,
                        "pickup_lng": 113.0,
                        "active_start_minute": current_minute + 10 + 12 * 60,
                        "active_end_minute": current_minute + 10 + 12 * 60 + 120,
                    },
                }
            ]
        )
        memory = MarketMemory()
        for offset in range(4):
            memory.remember_query(
                driver_id="D001",
                current_minute=8 * 60 + offset,
                current_lat=23.0,
                current_lng=113.0,
                raw_items=[],
                reachable_items=[],
                market_sample={"reachable_count": 0, "profitable_count": 0, "best_net": 0.0, "score": -10000.0},
            )
        for offset in range(4):
            memory.remember_query(
                driver_id="D001",
                current_minute=9 * 60 + offset,
                current_lat=23.0,
                current_lng=113.0,
                raw_items=[{}],
                reachable_items=[{}],
                market_sample={"reachable_count": 1, "profitable_count": 1, "best_net": 300.0, "score": 300.0},
            )
        service._market_memory = memory  # type: ignore[assignment]

        action = service.decide("D001")

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 10}})
        self.assertEqual(api.model_calls, 0)

    def test_clear_best_reachable_cargo_is_taken_without_model_call(self) -> None:
        api = FakeApi()
        load_time = ["2026-03-01 00:00:00", "2026-03-01 23:59:59"]
        api.items = [
            item(
                f"cheap_{idx}",
                remove_time="2026-03-01 23:59:59",
                load_time=load_time,
                price=120.0,
                distance_km=1.0,
            )
            for idx in range(25)
        ]
        api.items.append(
            item(
                "best",
                remove_time="2026-03-01 23:59:59",
                load_time=load_time,
                price=2000.0,
                distance_km=25.0,
            )
        )
        service = ModelDecisionService(api)  # type: ignore[arg-type]

        action = service.decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "best"}})
        self.assertEqual(api.model_calls, 0)


if __name__ == "__main__":
    unittest.main()
