from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.unknown_preference_guard import UnknownPreferenceGuard


class FakeApi:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("no fake response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        return {"choices": [{"message": {"content": content}}]}


def high_risk_unknown(content: str = "这条高罚偏好没识别出来。") -> dict[str, Any]:
    return {
        "rule_type": "unknown",
        "params": {"reason": "compile_failed"},
        "source_content": content,
        "penalty_amount": 5000,
        "penalty_cap": 10000,
        "risk_level": "high",
        "compile_error": "compile_failed",
    }


def low_risk_unknown() -> dict[str, Any]:
    rule = high_risk_unknown()
    rule["penalty_amount"] = 10
    rule["penalty_cap"] = 100
    rule["risk_level"] = "low"
    return rule


def unknown_batch() -> list[dict[str, Any]]:
    return [
        high_risk_unknown("如果路上感觉不顺，今天就别让我接任何活。"),
        high_risk_unknown("客户看起来不好说话的单我不想碰。"),
        high_risk_unknown("天气糟糕的时候别安排我出车。"),
        high_risk_unknown("太远或太近都不合适，你自己判断别让我吃亏。"),
        high_risk_unknown("平台口碑差的货主都别接。"),
        high_risk_unknown("接单前先等我电话确认。"),
        high_risk_unknown("月底前必须接那票熟货。"),
        high_risk_unknown("明天去那个仓库办事，别排冲突的活。"),
        high_risk_unknown("油价涨太多时别接远单。"),
        high_risk_unknown("外面说查得严的时候别往那边去。"),
    ]


def cargo_item(cargo_id: str, cargo_name: str = "普通货物") -> dict[str, Any]:
    return {
        "distance_km": 5.0,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": cargo_name,
            "price": 100000,
            "start": {"city": "深圳", "lat": 22.6, "lng": 114.0},
            "end": {"city": "广州", "lat": 23.1, "lng": 113.3},
            "load_time": ["2026-03-01 08:00:00", "2026-03-01 10:00:00"],
        },
    }


class UnknownPreferenceGuardTest(unittest.TestCase):
    def test_low_risk_unknown_does_not_call_guard(self) -> None:
        api = FakeApi([])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        items = [cargo_item("C1")]

        filtered = guard.filter_items("D001", {}, [low_risk_unknown()], items, 0)

        self.assertEqual(filtered, items)
        self.assertEqual(api.calls, 0)

    def test_high_risk_unknown_filters_blocked_cargo(self) -> None:
        api = FakeApi([{"decision": "allow", "blocked_cargo_ids": ["C1"], "reason": "possible violation"}])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        items = [cargo_item("C1"), cargo_item("C2")]

        filtered = guard.filter_items("D001", {}, [high_risk_unknown()], items, 0)

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["C2"])
        self.assertTrue(api.payloads[0]["enable_thinking"])

    def test_high_risk_unknown_reviews_candidates_after_first_batch(self) -> None:
        api = FakeApi(
            [
                {"decision": "allow", "blocked_cargo_ids": [], "reason": "safe"},
                {"decision": "allow", "blocked_cargo_ids": [], "reason": "safe"},
            ]
        )
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        items = [cargo_item(f"C{i}") for i in range(61)]

        filtered = guard.filter_items("D001", {}, [high_risk_unknown()], items, 0)

        self.assertEqual(len(filtered), 61)
        self.assertEqual(api.calls, 2)

    def test_invalid_filter_response_blocks_all_candidates(self) -> None:
        api = FakeApi(["{not json"])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]

        filtered = guard.filter_items("D001", {}, [high_risk_unknown()], [cargo_item("C1")], 0)

        self.assertEqual(filtered, [])

    def test_filter_response_without_decision_blocks_all_candidates(self) -> None:
        api = FakeApi([{}])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]

        filtered = guard.filter_items("D001", {}, [high_risk_unknown()], [cargo_item("C1")], 0)

        self.assertEqual(filtered, [])

    def test_batch_of_high_risk_unknowns_blocks_uncertain_candidates(self) -> None:
        api = FakeApi([{"decision": "uncertain", "blocked_cargo_ids": [], "reason": "missing external facts"}])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]

        filtered = guard.filter_items("D001", {}, unknown_batch(), [cargo_item("C1"), cargo_item("C2")], 0)

        self.assertEqual(filtered, [])
        payload = json.loads(api.payloads[0]["messages"][1]["content"])
        self.assertEqual(len(payload["unknown_rules"]), 10)
        self.assertEqual(payload["task"], "filter_cargo_candidates")

    def test_guard_action_allows_safe_action(self) -> None:
        api = FakeApi([{"decision": "allow", "blocked_cargo_ids": [], "reason": "safe"}])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        action = {"action": "take_order", "params": {"cargo_id": "C1"}}

        guarded = guard.guard_action("D001", {}, [high_risk_unknown()], action, 0, [cargo_item("C1")])

        self.assertEqual(guarded, action)

    def test_invalid_action_guard_response_waits(self) -> None:
        api = FakeApi(["{not json"])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        action = {"action": "reposition", "params": {"latitude": 23.0, "longitude": 113.0}}

        guarded = guard.guard_action("D001", {}, [high_risk_unknown()], action, 0, [])

        self.assertEqual(guarded, {"action": "wait", "params": {"duration_minutes": 15}})

    def test_batch_of_high_risk_unknowns_waits_on_uncertain_action(self) -> None:
        api = FakeApi([{"decision": "uncertain", "blocked_cargo_ids": [], "wait_minutes": 3, "reason": "cannot verify"}])
        guard = UnknownPreferenceGuard(api)  # type: ignore[arg-type]
        action = {"action": "take_order", "params": {"cargo_id": "C1"}}

        guarded = guard.guard_action("D001", {}, unknown_batch(), action, 0, [cargo_item("C1")])

        self.assertEqual(guarded, {"action": "wait", "params": {"duration_minutes": 15}})
        payload = json.loads(api.payloads[0]["messages"][1]["content"])
        self.assertEqual(len(payload["unknown_rules"]), 10)
        self.assertEqual(payload["task"], "guard_action")


if __name__ == "__main__":
    unittest.main()
