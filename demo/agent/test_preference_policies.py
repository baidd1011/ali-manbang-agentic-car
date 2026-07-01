from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.preference_policies import PreferencePolicyEngine


class FakeApi:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]:
        return {"driver_id": driver_id, "records": self.records}


def month_deadhead_rule() -> dict[str, Any]:
    return {
        "rule_type": "distance_limit",
        "params": {"metric": "month_deadhead", "max_km": 100},
        "penalty_amount": 10,
        "penalty_cap": 2000,
    }


def quiet_window_rule() -> dict[str, Any]:
    return {
        "rule_type": "quiet_window",
        "params": {"start_minute": 0, "end_minute": 360, "crosses_midnight": False},
        "penalty_amount": 1800,
        "penalty_cap": 55800,
    }


def cargo_item(
    pickup_km: float,
    cargo_id: str = "C1",
    price: int = 1000,
    cargo_name: str = "普通货物",
    start_city: str = "广东省广州市增城区",
    end_city: str = "广东省佛山市顺德区",
    end_lat: float = 23.0,
    end_lng: float = 113.0,
    finish_minutes: int | None = None,
) -> dict[str, Any]:
    item = {
        "distance_km": pickup_km,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": cargo_name,
            "price": price,
            "start": {"city": start_city, "lat": 23.0, "lng": 113.0},
            "end": {"city": end_city, "lat": end_lat, "lng": end_lng},
        },
    }
    if finish_minutes is not None:
        item["finish_minutes"] = finish_minutes
    return item


def old_cargo_item(pickup_km: float, cargo_id: str = "C1", price: int = 1000, cargo_name: str = "普通货物") -> dict[str, Any]:
    return {
        "distance_km": pickup_km,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": cargo_name,
            "price": price,
            "start": {"lat": 23.0, "lng": 113.0},
            "end": {"lat": 23.0, "lng": 113.0},
        },
    }


def active_order_records(days: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prev_end = 0
    for day in days:
        action_start = day * 1440 + 60
        records.append(
            {
                "query_scan_cost_minutes": action_start - prev_end,
                "action_exec_cost_minutes": 1,
                "position_before": {"lat": 23.0, "lng": 113.0},
                "position_after": {"lat": 23.0, "lng": 113.1},
                "action": {"action": "take_order"},
                "result": {"accepted": True, "simulation_progress_minutes": action_start + 1},
            }
        )
        prev_end = action_start + 1
    return records


class PreferencePolicyDistanceTest(unittest.TestCase):
    def test_month_deadhead_blocks_any_pickup_that_crosses_cap(self) -> None:
        records = [
            {
                "action": {"action": "take_order"},
                "result": {"accepted": True, "pickup_deadhead_km": 99.5, "simulation_progress_minutes": 10},
            }
        ]
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [month_deadhead_rule()]

        filtered = engine.filter_items("D003", {}, rules, [cargo_item(2.0), cargo_item(21.0)], 0)

        self.assertEqual(filtered, [])

    def test_month_deadhead_blocks_when_already_over_cap(self) -> None:
        records = [
            {
                "action": {"action": "take_order"},
                "result": {"accepted": True, "pickup_deadhead_km": 149.0, "simulation_progress_minutes": 10},
            }
        ]
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [month_deadhead_rule()]

        filtered = engine.filter_items("D003", {}, rules, [cargo_item(1.0), cargo_item(2.0)], 0)

        self.assertEqual(filtered, [])

    def test_month_deadhead_guard_blocks_take_order_that_crosses_cap(self) -> None:
        records = [
            {
                "action": {"action": "take_order"},
                "result": {"accepted": True, "pickup_deadhead_km": 99.5, "simulation_progress_minutes": 10},
            }
        ]
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [month_deadhead_rule()]
        action = {"action": "take_order", "params": {"cargo_id": "C1"}}

        guarded = engine.guard_action("D003", {}, rules, action, 0, [cargo_item(1.0)])

        self.assertEqual(guarded, {"action": "wait", "params": {"duration_minutes": 1}})

    def test_soft_rejected_category_is_avoided_when_alternative_exists(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "reject_cargo_category",
                "params": {"categories": ["食品饮料"], "hard": False},
                "penalty_amount": 200,
                "penalty_cap": 2000,
            }
        ]

        filtered = engine.filter_items(
            "D008",
            {},
            rules,
            [cargo_item(1.0, "food", cargo_name="食品饮料"), cargo_item(2.0, "normal", cargo_name="建材")],
            0,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["normal"])

    def test_quiet_window_conflict_is_filtered_even_when_profitable(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [quiet_window_rule()]
        items = [
            cargo_item(1.0, "cross", price=100000, finish_minutes=24 * 60 + 30),
            cargo_item(1.0, "clear", price=1000, finish_minutes=23 * 60 + 50),
        ]

        filtered = engine.filter_items("D002", {}, rules, items, 23 * 60)

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["clear"])

    def test_soft_rejected_category_kept_when_all_candidates_match(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "reject_cargo_category",
                "params": {"categories": ["食品饮料"], "hard": False},
                "penalty_amount": 200,
                "penalty_cap": 2000,
            }
        ]

        filtered = engine.filter_items(
            "D008",
            {},
            rules,
            [cargo_item(1.0, "food1", cargo_name="食品饮料"), cargo_item(2.0, "food2", cargo_name="食品饮料")],
            0,
        )

        self.assertEqual(len(filtered), 2)

    def test_city_filter_removes_pickup_or_dropoff_city(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "cargo_city_filter",
                "params": {"city_keywords": ["惠州"], "applies_to": "pickup_or_dropoff", "hard": True},
            }
        ]

        filtered = engine.filter_items(
            "D001",
            {},
            rules,
            [
                cargo_item(1.0, "start_hz", start_city="广东省惠州市博罗县", end_city="广东省佛山市顺德区"),
                cargo_item(1.0, "end_hz", start_city="广东省东莞市长安镇", end_city="广东省惠州市惠东县"),
                cargo_item(1.0, "ok", start_city="广东省东莞市长安镇", end_city="广东省佛山市顺德区"),
            ],
            0,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["ok"])

    def test_city_filter_active_dates_only_apply_when_order_overlaps_date(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "cargo_city_filter",
                "params": {
                    "city_keywords": ["深圳"],
                    "applies_to": "pickup_or_dropoff",
                    "hard": True,
                    "active_ranges": [{"start_minute": 3 * 1440, "end_minute": 4 * 1440}],
                },
            }
        ]

        filtered_before = engine.filter_items(
            "D001",
            {},
            rules,
            [cargo_item(1.0, "sz", start_city="广东省深圳市南山区", finish_minutes=2 * 1440 + 600)],
            2 * 1440 + 500,
        )
        filtered_during = engine.filter_items(
            "D001",
            {},
            rules,
            [cargo_item(1.0, "sz", start_city="广东省深圳市南山区", finish_minutes=3 * 1440 + 600)],
            3 * 1440 + 500,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered_before], ["sz"])
        self.assertEqual(filtered_during, [])

    def test_location_bounds_checks_pickup_and_dropoff_points(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "location_bounds",
                "params": {"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66},
            }
        ]
        outside_pickup = {
            "distance_km": 1.0,
            "cargo": {
                "cargo_id": "outside_pickup",
                "cargo_name": "普通货物",
                "price": 1000,
                "start": {"lat": 23.0, "lng": 113.0},
                "end": {"lat": 22.6, "lng": 114.0},
            },
        }
        inside = {
            "distance_km": 1.0,
            "cargo": {
                "cargo_id": "inside",
                "cargo_name": "普通货物",
                "price": 1000,
                "start": {"lat": 22.5, "lng": 114.0},
                "end": {"lat": 22.6, "lng": 114.1},
            },
        }

        filtered = engine.filter_items("D003", {}, rules, [outside_pickup, inside], 0)

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["inside"])

    def test_daily_rest_conflict_is_filtered_even_when_profitable(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "daily_rest",
                "params": {"minutes": 8 * 60, "required_count": 1, "applies_on": "daily"},
                "penalty_amount": 100,
                "penalty_cap": 1000,
            }
        ]
        items = [
            cargo_item(1.0, "conflict", price=100000, finish_minutes=24 * 60 + 22 * 60),
            cargo_item(1.0, "clear", price=1000, finish_minutes=24 * 60 + 12 * 60),
        ]

        filtered = engine.filter_items("D001", {}, rules, items, 20 * 60)

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["clear"])

    def test_daily_home_without_quiet_window_only_enforces_deadline(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "daily_home",
                "params": {"home_lat": 23.0, "home_lng": 113.0, "radius_km": 1, "deadline_minute": 22 * 60},
            }
        ]

        action = engine.pre_query_action(
            "D006",
            {"simulation_progress_minutes": 21 * 60 + 40, "current_lat": 23.5, "current_lng": 113.5},
            rules,
        )
        filtered = engine.filter_items(
            "D006",
            {},
            rules,
            [cargo_item(1.0, "late", end_lat=23.5, end_lng=113.5, finish_minutes=21 * 60 + 55)],
            21 * 60 + 30,
        )

        self.assertEqual(action["action"], "reposition")
        self.assertEqual(filtered, [])

    def test_daily_home_non_crossing_quiet_window_only_blocks_inside_window(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "daily_home",
                "params": {
                    "home_lat": 23.0,
                    "home_lng": 113.0,
                    "radius_km": 1,
                    "deadline_minute": 0,
                    "quiet_start_minute": 0,
                    "quiet_end_minute": 7 * 60,
                },
            }
        ]

        no_action = engine.pre_query_action(
            "D006",
            {"simulation_progress_minutes": 23 * 60, "current_lat": 23.0, "current_lng": 113.0},
            rules,
        )
        wait_action = engine.pre_query_action(
            "D006",
            {"simulation_progress_minutes": 60, "current_lat": 23.0, "current_lng": 113.0},
            rules,
        )

        self.assertIsNone(no_action)
        self.assertEqual(wait_action, {"action": "wait", "params": {"duration_minutes": 6 * 60}})

    def test_city_day_goal_prefers_matching_city_when_behind(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "cargo_city_day_goal",
                "params": {
                    "city_keywords": ["增城"],
                    "applies_to": "pickup_or_dropoff",
                    "required_days": 4,
                    "target_lat": 23.15,
                    "target_lng": 113.67,
                    "radius_km": 1,
                },
            }
        ]

        filtered = engine.filter_items(
            "D002",
            {},
            rules,
            [
                cargo_item(1.0, "zc", start_city="广东省广州市增城区", end_city="广东省佛山市顺德区"),
                cargo_item(1.0, "normal", start_city="广东省东莞市长安镇", end_city="广东省佛山市顺德区"),
            ],
            8 * 1440,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["zc"])

    def test_scheduled_stop_resolves_city_goal_point_and_repositions(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "cargo_city_day_goal",
                "params": {
                    "city_keywords": ["增城"],
                    "applies_to": "pickup_or_dropoff",
                    "required_days": 4,
                    "target_lat": 23.15,
                    "target_lng": 113.67,
                    "radius_km": 1,
                },
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 11 * 1440,
                    "active_end_minute": 12 * 1440 - 1,
                    "stops": [
                        {"name": "增城老档口", "place_keywords": ["增城", "档口"], "stay_minutes": 120, "radius_km": 1}
                    ],
                },
            },
        ]
        status = {"simulation_progress_minutes": 11 * 1440 + 360, "current_lat": 23.56, "current_lng": 116.3}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.15, "longitude": 113.67}})

    def test_unresolved_active_scheduled_stop_waits_conservatively(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "point_visit",
                "params": {"target_lat": 23.13, "target_lng": 113.26, "radius_km": 1, "required_days": 5},
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 10 * 1440,
                    "active_end_minute": 11 * 1440 - 1,
                    "stops": [{"radius_km": 1}],
                },
            },
        ]
        status = {"simulation_progress_minutes": 10 * 1440 + 60, "current_lat": 22.8, "current_lng": 114.2}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 15}})

    def test_unrelated_point_visit_does_not_resolve_scheduled_stop(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "point_visit",
                "params": {"target_lat": 23.13, "target_lng": 113.26, "radius_km": 1, "required_days": 5},
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 10 * 1440,
                    "active_end_minute": 11 * 1440 - 1,
                    "stops": [{"name": "医院", "place_keywords": ["医院"], "radius_km": 1}],
                },
            },
        ]
        status = {"simulation_progress_minutes": 10 * 1440 + 60, "current_lat": 22.8, "current_lng": 114.2}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 15}})

    def test_scheduled_reposition_waits_if_it_would_cross_quiet_window(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        active_start = 11 * 1440
        rules = [
            quiet_window_rule(),
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 12 * 1440 - 1,
                    "stops": [{"name": "增城", "lat": 23.15, "lng": 113.67, "stay_minutes": 120, "radius_km": 1}],
                },
            },
        ]
        status = {"simulation_progress_minutes": active_start - 240, "current_lat": 21.86, "current_lng": 111.19}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 240}})

    def test_route_event_prepares_at_previous_quiet_end_for_morning_deadline(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        active_start = 30 * 1440
        rules = [
            quiet_window_rule(),
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {
                            "name": "四会",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": active_start + 720,
                            "stay_until_minute": active_start + 840,
                            "radius_km": 1,
                        },
                    ],
                },
            },
        ]
        status = {"simulation_progress_minutes": active_start - 18 * 60, "current_lat": 23.54, "current_lng": 116.74}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.15, "longitude": 113.67}})

    def test_prepositioned_route_stop_waits_until_event_day(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        active_start = 30 * 1440
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {"name": "四会", "lat": 23.32, "lng": 112.83, "deadline_minute": active_start + 720, "radius_km": 1},
                    ],
                },
            }
        ]
        status = {"simulation_progress_minutes": active_start - 12 * 60, "current_lat": 23.15, "current_lng": 113.67}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 720}})

    def test_no_stay_stop_done_advances_to_next_stop(self) -> None:
        records = [
            {
                "query_scan_cost_minutes": 0,
                "action_exec_cost_minutes": 100,
                "position_before": {"lat": 22.0, "lng": 113.0},
                "position_after": {"lat": 23.15, "lng": 113.67},
                "action": {"action": "reposition"},
                "result": {"simulation_progress_minutes": 100},
            }
        ]
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 0,
                    "active_end_minute": 1000,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {"name": "四会", "lat": 23.32, "lng": 112.83, "deadline_minute": 500, "radius_km": 1},
                    ],
                },
            }
        ]
        status = {"simulation_progress_minutes": 100, "current_lat": 23.15, "current_lng": 113.67}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.32, "longitude": 112.83}})

    def test_city_goal_repositions_when_behind_schedule(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "cargo_city_day_goal",
                "params": {
                    "city_keywords": ["增城"],
                    "applies_to": "pickup_or_dropoff",
                    "required_days": 4,
                    "target_lat": 23.15,
                    "target_lng": 113.67,
                    "radius_km": 1,
                },
            }
        ]
        status = {"simulation_progress_minutes": 24 * 1440, "current_lat": 22.8, "current_lng": 114.2}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.15, "longitude": 113.67}})

    def test_scheduled_stop_waits_when_at_target(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 11 * 1440,
                    "active_end_minute": 12 * 1440 - 1,
                    "stops": [{"name": "增城", "lat": 23.15, "lng": 113.67, "stay_minutes": 120, "radius_km": 1}],
                },
            }
        ]
        status = {"simulation_progress_minutes": 11 * 1440 + 360, "current_lat": 23.15, "current_lng": 113.67}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 120}})

    def test_scheduled_stop_with_stay_until_is_not_done_on_arrival(self) -> None:
        active_start = 30 * 1440
        records = [
            {
                "query_scan_cost_minutes": 0,
                "action_exec_cost_minutes": 88,
                "position_before": {"lat": 23.15, "lng": 113.67},
                "position_after": {"lat": 23.32, "lng": 112.83},
                "action": {"action": "reposition"},
                "result": {"simulation_progress_minutes": active_start + 448},
            }
        ]
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {
                            "name": "四会",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": active_start + 720,
                            "stay_until_minute": active_start + 840,
                            "radius_km": 1,
                        }
                    ],
                },
            }
        ]
        status = {"simulation_progress_minutes": active_start + 448, "current_lat": 23.32, "current_lng": 112.83}

        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 392}})

    def test_scheduled_event_filters_order_that_blocks_deadline(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 30 * 1440,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {
                            "name": "四会县城",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": 30 * 1440 + 720,
                            "stay_until_minute": 30 * 1440 + 840,
                            "radius_km": 1,
                        }
                    ],
                },
            }
        ]

        filtered = engine.filter_items(
            "D002",
            {},
            rules,
            [
                cargo_item(1.0, "late", end_lat=22.0, end_lng=114.5, finish_minutes=30 * 1440 + 700),
                cargo_item(1.0, "ok", end_lat=23.32, end_lng=112.83, finish_minutes=30 * 1440 + 650),
            ],
            30 * 1440 + 360,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["ok"])

    def test_unresolved_scheduled_event_filters_activity_crossing_prepare_window(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        active_start = 10 * 1440
        rules = [
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 11 * 1440 - 1,
                    "stops": [{"name": "医院", "place_keywords": ["医院"], "radius_km": 1}],
                },
            }
        ]

        filtered = engine.filter_items(
            "D002",
            {},
            rules,
            [cargo_item(1.0, "cross", finish_minutes=active_start - 100)],
            active_start - 13 * 60,
        )

        self.assertEqual(filtered, [])

    def test_month_rest_visible_preference_plans_before_month_end(self) -> None:
        rules = [{"rule_type": "month_day_rest", "params": {"required_days": 2, "mode": "no_active"}}]

        early_action = PreferencePolicyEngine(FakeApi([])).pre_query_action(
            "D002",
            {"simulation_progress_minutes": 0, "current_lat": 23.0, "current_lng": 113.0},
            rules,
        )
        planned_day_action = PreferencePolicyEngine(FakeApi(active_order_records(list(range(9))))).pre_query_action(
            "D002",
            {"simulation_progress_minutes": 9 * 1440, "current_lat": 23.0, "current_lng": 113.0},
            rules,
        )

        self.assertIsNone(early_action)
        self.assertEqual(planned_day_action, {"action": "wait", "params": {"duration_minutes": 1440}})

    def test_month_rest_planned_day_filters_crossing_cargo(self) -> None:
        engine = PreferencePolicyEngine(FakeApi(active_order_records(list(range(8)))))  # type: ignore[arg-type]
        rules = [{"rule_type": "month_day_rest", "params": {"required_days": 2, "mode": "no_active"}}]
        current_minute = 8 * 1440 + 12 * 60

        filtered = engine.filter_items(
            "D002",
            {},
            rules,
            [
                cargo_item(1.0, "same_day_ok", finish_minutes=8 * 1440 + 20 * 60),
                cargo_item(1.0, "cross_rest_day", finish_minutes=9 * 1440 + 60),
            ],
            current_minute,
        )

        self.assertEqual([item["cargo"]["cargo_id"] for item in filtered], ["same_day_ok"])

    def test_month_rest_fallback_skips_scheduled_prepare_day(self) -> None:
        records: list[dict[str, Any]] = []
        prev_end = 0
        for day in range(27):
            action_start = day * 1440 + 60
            records.append(
                {
                    "query_scan_cost_minutes": action_start - prev_end,
                    "action_exec_cost_minutes": 1,
                    "position_before": {"lat": 23.0, "lng": 113.0},
                    "position_after": {"lat": 23.0, "lng": 113.1},
                    "action": {"action": "take_order"},
                    "result": {"accepted": True, "simulation_progress_minutes": action_start + 1},
                }
            )
            prev_end = action_start + 1
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        active_start = 30 * 1440
        rules = [
            quiet_window_rule(),
            {
                "rule_type": "month_day_rest",
                "required_days": 2,
                "mode": "no_active",
                "params": {"required_days": 2, "mode": "no_active"},
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {
                            "name": "四会",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": active_start + 720,
                            "stay_until_minute": active_start + 840,
                            "radius_km": 1,
                        },
                    ],
                },
            },
        ]

        wait = engine.month_day_market_wait_minutes(
            "D002",
            {"simulation_progress_minutes": 27 * 1440 + 360, "current_lat": 23.0, "current_lng": 113.0},
            rules,
            27 * 1440 + 360,
            {"score": 9999.0, "reachable_count": 100, "profitable_count": 100},
        )

        self.assertEqual(wait, 1080)

    def test_month_rest_fallback_skips_daily_home_takeover(self) -> None:
        engine = PreferencePolicyEngine(FakeApi([]))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "month_day_rest",
                "params": {"required_days": 1, "mode": "no_active"},
            },
            {
                "rule_type": "daily_home",
                "params": {
                    "home_lat": 23.02,
                    "home_lng": 113.75,
                    "radius_km": 1,
                    "deadline_minute": 23 * 60,
                },
            },
        ]
        current_minute = 30 * 1440 + 22 * 60 + 30
        status = {"simulation_progress_minutes": current_minute, "current_lat": 22.0, "current_lng": 114.5}

        wait = engine.month_day_market_wait_minutes(
            "D002",
            status,
            rules,
            current_minute,
            {"score": 9999.0, "reachable_count": 100, "profitable_count": 100},
        )
        action = engine.pre_query_action("D002", status, rules)

        self.assertEqual(wait, 0)
        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.02, "longitude": 113.75}})

    def test_scheduled_prepare_preempts_protected_month_rest_day(self) -> None:
        records: list[dict[str, Any]] = []
        prev_end = 0
        for day in range(29):
            action_start = day * 1440 + 60
            records.append(
                {
                    "query_scan_cost_minutes": action_start - prev_end,
                    "action_exec_cost_minutes": 1,
                    "position_before": {"lat": 23.0, "lng": 113.0},
                    "position_after": {"lat": 23.0, "lng": 113.1},
                    "action": {"action": "take_order"},
                    "result": {"accepted": True, "simulation_progress_minutes": action_start + 1},
                }
            )
            prev_end = action_start + 1
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        active_start = 30 * 1440
        rules = [
            quiet_window_rule(),
            {
                "rule_type": "month_day_rest",
                "params": {"required_days": 2, "mode": "no_active"},
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {
                            "name": "四会",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": active_start + 720,
                            "stay_until_minute": active_start + 840,
                            "radius_km": 1,
                        },
                    ],
                },
            },
        ]

        action = engine.pre_query_action(
            "D002",
            {"simulation_progress_minutes": 29 * 1440 + 360, "current_lat": 22.63, "current_lng": 114.22},
            rules,
        )

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.15, "longitude": 113.67}})

    def test_scheduled_prepare_preempts_city_goal_and_month_rest(self) -> None:
        records: list[dict[str, Any]] = []
        prev_end = 0
        for day in range(29):
            action_start = day * 1440 + 60
            records.append(
                {
                    "query_scan_cost_minutes": action_start - prev_end,
                    "action_exec_cost_minutes": 1,
                    "position_before": {"lat": 23.0, "lng": 113.0},
                    "position_after": {"lat": 23.0, "lng": 113.1},
                    "action": {"action": "take_order"},
                    "result": {"accepted": True, "simulation_progress_minutes": action_start + 1},
                }
            )
            prev_end = action_start + 1
        engine = PreferencePolicyEngine(FakeApi(records))  # type: ignore[arg-type]
        rules = [
            {
                "rule_type": "month_day_rest",
                "params": {"required_days": 1, "mode": "no_active"},
            },
            {
                "rule_type": "cargo_city_day_goal",
                "params": {
                    "city_keywords": ["增城"],
                    "applies_to": "pickup_or_dropoff",
                    "required_days": 4,
                    "target_lat": 23.15,
                    "target_lng": 113.67,
                    "radius_km": 1,
                },
            },
            {
                "rule_type": "scheduled_event",
                "params": {
                    "mode": "stops",
                    "active_start_minute": 30 * 1440,
                    "active_end_minute": 31 * 1440 - 1,
                    "stops": [
                        {"name": "增城", "lat": 23.15, "lng": 113.67, "radius_km": 1},
                        {
                            "name": "四会",
                            "lat": 23.32,
                            "lng": 112.83,
                            "deadline_minute": 30 * 1440 + 720,
                            "stay_until_minute": 30 * 1440 + 840,
                            "radius_km": 1,
                        },
                    ],
                },
            },
        ]

        action = engine.pre_query_action(
            "D002",
            {"simulation_progress_minutes": 29 * 1440 + 360, "current_lat": 22.8, "current_lng": 114.2},
            rules,
        )

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 23.15, "longitude": 113.67}})


if __name__ == "__main__":
    unittest.main()
