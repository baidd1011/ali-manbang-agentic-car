from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.preference_compiler import PreferenceCompiler


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


def pref(content: str, penalty_amount: int = 100, penalty_cap: int | None = 1000) -> dict[str, Any]:
    return {
        "content": content,
        "start_time": "2026-03-01 00:00:00",
        "end_time": "2026-03-31 23:59:59",
        "penalty_amount": penalty_amount,
        "penalty_cap": penalty_cap,
    }


def llm_pair(rule_type: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"rule_type": rule_type, "confidence": 0.95}, {"params": params}]


class PreferenceCompilerTest(unittest.TestCase):
    def test_cache_reuses_compiled_rule(self) -> None:
        api = FakeApi(llm_pair("reject_cargo_category", {"categories": ["蔬菜"], "hard": True}))
        compiler = PreferenceCompiler(api)
        preference = pref("不接货源品类为蔬菜的订单。")

        first = compiler.compile("D002", [preference])
        second = compiler.compile("D002", [preference])

        self.assertEqual(api.calls, 2)
        self.assertTrue(all(payload.get("temperature") == 0 for payload in api.payloads))
        self.assertEqual(first[0]["rule_type"], "reject_cargo_category")
        self.assertEqual(second[0]["params"]["categories"], ["蔬菜"])
        self.assertEqual(first[0]["compile_status"], "compiled")
        self.assertEqual(first[0]["risk_level"], "medium")
        self.assertAlmostEqual(first[0]["compile_confidence"], 0.95)

    def test_high_risk_preference_uses_thinking_first(self) -> None:
        api = FakeApi(llm_pair("daily_rest", {"hours": 8, "required_count": 1, "applies_on": "daily"}))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("每天得睡满八小时。", penalty_amount=10000, penalty_cap=10000)])

        self.assertEqual(api.calls, 2)
        self.assertTrue(all(payload["enable_thinking"] for payload in api.payloads))
        self.assertEqual(rules[0]["risk_level"], "high")
        self.assertEqual(rules[0]["compile_status"], "thinking_compiled")

    def test_fixed_sleep_window_normalizes_to_quiet_window(self) -> None:
        api = FakeApi(llm_pair("quiet_window", {"start_clock": "00:00", "end_clock": "06:00", "crosses_midnight": False}))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile(
            "D002",
            [pref("零点以后到早上六点这段我得睡觉，车得停着熄火，雷打不动。", penalty_amount=1800, penalty_cap=55800)],
        )

        self.assertEqual(rules[0]["rule_type"], "quiet_window")
        self.assertEqual(rules[0]["params"]["start_minute"], 0)
        self.assertEqual(rules[0]["params"]["end_minute"], 360)

    def test_city_filter_accepts_active_dates_and_string_keyword(self) -> None:
        api = FakeApi(
            llm_pair(
                "cargo_city_filter",
                {
                    "city_keywords": "深圳",
                    "applies_to": "pickup_or_dropoff",
                    "hard": True,
                    "active_dates": ["2026-03-04", "2026-03-05"],
                },
            )
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("三月四号五号交警在深圳查车，这两天我不往深圳跑，也别派进深圳的货。")])

        self.assertEqual(rules[0]["params"]["city_keywords"], ["深圳"])
        self.assertEqual(
            rules[0]["params"]["active_ranges"],
            [{"start_minute": 3 * 1440, "end_minute": 4 * 1440}, {"start_minute": 4 * 1440, "end_minute": 5 * 1440}],
        )

    def test_daily_home_allows_missing_quiet_window(self) -> None:
        api = FakeApi(llm_pair("daily_home", {"home": {"lat": 23.40, "lng": 113.16}, "radius_km": 2, "deadline_clock": "22:00"}))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D006", [pref("每天晚上十点之前到家，家在（23.40，113.16），两公里内算到。")])

        self.assertEqual(rules[0]["rule_type"], "daily_home")
        self.assertNotIn("quiet_start_minute", rules[0]["params"])
        self.assertEqual(rules[0]["params"]["deadline_minute"], 22 * 60)

    def test_month_day_rest_defaults_to_no_active(self) -> None:
        api = FakeApi(llm_pair("month_day_rest", {"required_days": 2}))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D002", [pref("这月得留两个整天停驶检修，那天别给我排活。")])

        self.assertEqual(rules[0]["params"]["mode"], "no_active")

    def test_unknown_classification_retries_before_caching_unknown(self) -> None:
        api = FakeApi(
            [
                {"rule_type": "unknown", "confidence": 0.9},
                *llm_pair("daily_rest", {"hours": 8, "required_count": 1, "applies_on": "daily"}),
            ]
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("我这人熬不住连轴转，每天至少连续停车熄火休息满8小时。")])

        self.assertEqual(api.calls, 3)
        self.assertEqual(rules[0]["rule_type"], "daily_rest")
        self.assertEqual(rules[0]["params"]["minutes"], 480)

    def test_retry_after_invalid_classification_json(self) -> None:
        api = FakeApi(
            [
                "{not json",
                *llm_pair("daily_rest", {"hours": 8, "required_count": 1, "applies_on": "daily"}),
            ]
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("每天连续停车休息满8小时。")])

        self.assertEqual(api.calls, 3)
        self.assertEqual(rules[0]["rule_type"], "daily_rest")
        self.assertEqual(rules[0]["params"]["minutes"], 480)

    def test_unknown_after_repeated_invalid_rules(self) -> None:
        api = FakeApi(
            [
                {"rule_type": "bad_type", "confidence": 0.9},
                {"rule_type": "daily_rest", "confidence": 0.9},
                {"params": {"minutes": -1}},
            ]
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("无法理解的偏好。")])

        self.assertEqual(api.calls, 4)
        self.assertTrue(api.payloads[-1]["enable_thinking"])
        self.assertEqual(rules[0]["rule_type"], "unknown")
        self.assertIn("无法理解", rules[0]["source_content"])

    def test_batch_of_unobservable_preferences_compile_to_high_risk_unknown(self) -> None:
        preferences = [
            pref("如果路上感觉不顺，今天就别让我接任何活。", penalty_amount=5000, penalty_cap=10000),
            pref("客户看起来不好说话的单我不想碰。", penalty_amount=5000, penalty_cap=10000),
            pref("天气糟糕的时候别安排我出车。", penalty_amount=5000, penalty_cap=10000),
            pref("太远或太近都不合适，你自己判断别让我吃亏。", penalty_amount=5000, penalty_cap=10000),
            pref("平台口碑差的货主都别接。", penalty_amount=5000, penalty_cap=10000),
            pref("接单前先等我电话确认。", penalty_amount=5000, penalty_cap=10000),
            pref("月底前必须接那票熟货。", penalty_amount=5000, penalty_cap=10000),
            pref("明天去那个仓库办事，别排冲突的活。", penalty_amount=5000, penalty_cap=10000),
            pref("油价涨太多时别接远单。", penalty_amount=5000, penalty_cap=10000),
            pref("外面说查得严的时候别往那边去。", penalty_amount=5000, penalty_cap=10000),
        ]
        api = FakeApi([{"rule_type": "unknown", "confidence": 0.92}] * 4 * len(preferences))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("DTEST", preferences)

        self.assertEqual(api.calls, 4 * len(preferences))
        self.assertTrue(all(rule["rule_type"] == "unknown" for rule in rules))
        self.assertTrue(all(rule["risk_level"] == "high" for rule in rules))
        self.assertTrue(all(rule["compile_status"] == "failed" for rule in rules))
        self.assertTrue(all(payload["enable_thinking"] for payload in api.payloads[0::4]))

    def test_low_penalty_unobservable_preferences_remain_low_risk_unknown(self) -> None:
        preferences = [
            pref("货主以前坑过人的单都不要。", penalty_amount=10, penalty_cap=100),
            pref("家里人说可以我再出车。", penalty_amount=10, penalty_cap=100),
        ]
        api = FakeApi([{"rule_type": "unknown", "confidence": 0.92}] * 3 * len(preferences))
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("DTEST", preferences)

        self.assertEqual(api.calls, 3 * len(preferences))
        self.assertTrue(all(rule["rule_type"] == "unknown" for rule in rules))
        self.assertTrue(all(rule["risk_level"] == "low" for rule in rules))
        self.assertTrue(all(not payload["enable_thinking"] for payload in api.payloads[0::3]))

    def test_thinking_fallback_can_recover_unknown(self) -> None:
        api = FakeApi(
            [
                {"rule_type": "unknown", "confidence": 0.9},
                {"rule_type": "unknown", "confidence": 0.9},
                {"rule_type": "cargo_city_filter", "confidence": 0.9},
                {"params": {"city_keywords": ["惠州"], "applies_to": "pickup_or_dropoff", "hard": True}},
            ]
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D001", [pref("装货地或卸货地在惠州的货，我一律不接。")])

        self.assertEqual(api.calls, 4)
        self.assertTrue(api.payloads[2]["enable_thinking"])
        self.assertTrue(api.payloads[3]["enable_thinking"])
        self.assertEqual(rules[0]["rule_type"], "cargo_city_filter")
        self.assertEqual(rules[0]["params"]["city_keywords"], ["惠州"])

    def test_must_take_cargo_uses_wall_time_not_model_minutes(self) -> None:
        preference = pref(
            "指定熟货源编号240646：装货地（24.81，113.58）；上架时间：2026-03-03 14:43:36。",
            penalty_amount=10000,
            penalty_cap=10000,
        )
        preference["end_time"] = "2026-03-03 16:08:24"
        api = FakeApi(
            llm_pair(
                "must_take_cargo",
                {
                    "cargo_id": "240646",
                    "pickup": {"lat": 24.81, "lng": 113.58},
                    "cargo_available_time": "2026-03-03 14:43:36",
                    "preference_end_time": "2026-03-03 16:08:24",
                },
            )
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D009", [preference])

        self.assertTrue(all(payload["enable_thinking"] for payload in api.payloads))
        self.assertEqual(rules[0]["params"]["active_start_minute"], 3763)
        self.assertEqual(rules[0]["params"]["active_end_minute"], 3848)

    def test_zero_coordinates_are_rejected(self) -> None:
        api = FakeApi(
            [
                {"rule_type": "daily_home", "confidence": 0.9},
                {"params": {"home": {"lat": 0, "lng": 0}, "deadline_clock": "23:00", "quiet_start_clock": "23:00", "quiet_end_clock": "06:00"}},
                {"rule_type": "daily_home", "confidence": 0.9},
                {"params": {"home": {"lat": 0, "lng": 0}, "deadline_clock": "23:00", "quiet_start_clock": "23:00", "quiet_end_clock": "06:00"}},
            ]
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D005", [pref("每天23点至次日6点不接单、不空车赶路。")])

        self.assertEqual(api.calls, 5)
        self.assertTrue(api.payloads[-1]["enable_thinking"])
        self.assertEqual(rules[0]["rule_type"], "unknown")

    def test_current_rule_type_shapes(self) -> None:
        pairs = [
            llm_pair("daily_rest", {"hours": 4, "required_count": 1, "applies_on": "weekday"}),
            llm_pair("quiet_window", {"start_clock": "23:00", "end_clock": "06:00", "crosses_midnight": True}),
            llm_pair("cargo_city_filter", {"city_keywords": ["惠州"], "applies_to": "pickup_or_dropoff", "hard": True}),
            llm_pair("cargo_city_day_goal", {"city_keywords": ["增城"], "applies_to": "pickup_or_dropoff", "required_days": 4, "target": {"lat": 23.15, "lng": 113.67}, "radius_km": 1}),
            llm_pair("distance_limit", {"metric": "haul", "max_km": 150}),
            llm_pair("distance_limit", {"metric": "pickup_deadhead", "max_km": 50}),
            llm_pair("distance_limit", {"metric": "month_deadhead", "max_km": 100}),
            llm_pair("month_day_rest", {"required_days": 2, "mode": "no_active"}),
            llm_pair("order_cadence", {"first_order_before_clock": "12:00"}),
            llm_pair("order_cadence", {"max_orders_per_day": 3}),
            llm_pair("location_bounds", {"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66}),
            llm_pair("location_exclusion_circle", {"center": {"lat": 23.30, "lng": 113.52}, "radius_km": 20}),
            llm_pair("daily_home", {"home": {"lat": 23.12, "lng": 113.28}, "radius_km": 1, "deadline_clock": "23:00", "quiet_start_clock": "23:00", "quiet_end_clock": "08:00"}),
            llm_pair("scheduled_event", {"event_start_time": "2026-03-10 10:00:00", "pickup": {"lat": 23.21, "lng": 113.37}, "home": {"lat": 23.19, "lng": 113.36}, "pickup_stay_minutes": 10, "home_deadline_time": "2026-03-10 22:00:00", "stay_until_time": "2026-03-13 22:00:00", "radius_km": 1}),
            llm_pair("point_visit", {"target": {"lat": 23.13, "lng": 113.26}, "radius_km": 1, "required_days": 5}),
        ]
        responses = [item for pair in pairs for item in pair]
        api = FakeApi(responses)
        compiler = PreferenceCompiler(api)
        preferences = [pref(f"preference {idx}") for idx in range(len(pairs))]

        rules = compiler.compile("DALL", preferences)

        self.assertEqual(api.calls, len(pairs) * 2)
        self.assertEqual(rules[0]["params"]["applies_on"], "weekday")
        self.assertEqual(rules[0]["params"]["minutes"], 240)
        self.assertEqual(rules[1]["params"]["start_minute"], 1380)
        self.assertEqual(rules[2]["params"]["city_keywords"], ["惠州"])
        self.assertEqual(rules[3]["params"]["target_lat"], 23.15)
        self.assertEqual(rules[-1]["params"]["required_days"], 5)

    def test_generic_scheduled_event_stops(self) -> None:
        api = FakeApi(
            llm_pair(
                "scheduled_event",
                {
                    "active_start_time": "2026-03-31 00:00:00",
                    "active_end_time": "2026-03-31 23:59:59",
                    "stops": [
                        {"name": "增城区档口", "place_keywords": ["增城", "增城区", "档口"], "radius_km": 1},
                        {
                            "name": "四会县城",
                            "target": {"lat": 23.32, "lng": 112.83},
                            "deadline_time": "2026-03-31 12:00:00",
                            "stay_until_time": "2026-03-31 14:00:00",
                            "radius_km": 1,
                        },
                    ],
                },
            )
        )
        compiler = PreferenceCompiler(api)

        rules = compiler.compile("D002", [pref("三月三十一号上午先过增城区档口，中午十二点前赶到四会县城赴宴到下午两点。")])

        self.assertEqual(rules[0]["rule_type"], "scheduled_event")
        self.assertEqual(rules[0]["params"]["mode"], "stops")
        self.assertEqual(rules[0]["params"]["active_start_minute"], 43200)
        self.assertEqual(rules[0]["params"]["stops"][0]["place_keywords"], ["增城", "增城区", "档口"])
        self.assertEqual(rules[0]["params"]["stops"][1]["deadline_minute"], 43920)


if __name__ == "__main__":
    unittest.main()
