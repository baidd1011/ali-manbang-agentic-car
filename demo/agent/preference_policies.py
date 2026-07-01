"""Policy execution for compiled driver preference rules."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_REPOSITION_SPEED_KM_PER_HOUR = 60.0
_COST_PER_KM = 1.5
_NO_FEASIBLE_CARGO_WAIT_MINUTES = 1
_UNRESOLVED_EVENT_WAIT_MINUTES = 15
_SIMULATION_DAYS = 31
_MONTHLY_REST_SCORING_DAYS = 30
_MARKET_HISTORY_MIN_SAMPLES = 6
_MARKET_HISTORY_MAX_SAMPLES = 200
_MARKET_LOW_QUANTILE = 0.25
_MARKET_TOP_NET_COUNT = 3


class PreferencePolicyEngine:
    """Apply compiled preference rules to actions and cargo candidates."""

    def __init__(self, api: SimulationApiPort, logger: logging.Logger | None = None) -> None:
        self._api = api
        self._logger = logger or logging.getLogger("agent.preference_policies")
        self._market_history_by_driver: dict[str, list[dict[str, Any]]] = {}
        self._city_goal_days_by_driver: dict[str, dict[str, set[int]]] = {}

    def pre_query_action(self, driver_id: str, status: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any] | None:
        for action in (
            self._scheduled_event_action(driver_id, status, rules),
            self._must_take_cargo_action(driver_id, status, rules),
            self._month_day_fallback_action(driver_id, status, rules),
            self._city_goal_action(driver_id, status, rules),
            self._daily_home_action(status, rules),
            self._monthly_visit_action(driver_id, status, rules),
        ):
            if action is not None:
                return action

        rest_minutes = self._daily_rest_wait_minutes(driver_id, status, rules)
        if rest_minutes > 0:
            self._logger.info("daily continuous rest required driver_id=%s wait_minutes=%s", driver_id, rest_minutes)
            return {"action": "wait", "params": {"duration_minutes": rest_minutes}}

        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        quiet_wait_minutes = self._quiet_window_wait_minutes(rules, current_minute)
        if quiet_wait_minutes > 0:
            self._logger.info("quiet time window required driver_id=%s wait_minutes=%s", driver_id, quiet_wait_minutes)
            return {"action": "wait", "params": {"duration_minutes": quiet_wait_minutes}}

        calendar_wait_minutes = self._calendar_preference_wait_minutes(driver_id, status, rules)
        if calendar_wait_minutes > 0:
            self._logger.info("calendar preference required driver_id=%s wait_minutes=%s", driver_id, calendar_wait_minutes)
            return {"action": "wait", "params": {"duration_minutes": calendar_wait_minutes}}
        return None

    def filter_items(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        items: list[dict[str, Any]],
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        filtered = self._filter_rejected_cargo_items(items, rules)
        filtered = self._filter_city_rule_items(filtered, rules, decision_time_min, hard=True)
        filtered = self._filter_daily_home_items(filtered, rules, driver_id, decision_time_min)
        filtered = self._filter_location_bounds_items(filtered, rules)
        filtered = self._filter_location_exclusion_items(filtered, rules)
        filtered = self._filter_distance_rule_items(filtered, rules, driver_id)
        filtered = self._filter_scheduled_event_conflict_items(filtered, rules, driver_id, decision_time_min)
        filtered = self._filter_rest_conflict_items(filtered, rules, decision_time_min)
        filtered = self._filter_quiet_window_conflict_items(filtered, rules, decision_time_min)
        filtered = self._filter_month_day_conflict_items(filtered, rules, driver_id, decision_time_min)
        filtered = self._filter_city_goal_items(filtered, rules, driver_id, decision_time_min)
        filtered = self._filter_city_rule_items(filtered, rules, decision_time_min, hard=False)
        return self._filter_soft_rejected_cargo_items(filtered, rules)

    def guard_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        action: dict[str, Any],
        decision_time_min: int,
        reachable_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        guarded = self._guard_reposition_quiet_window_action(action, status, rules, decision_time_min)
        guarded = self._guard_reposition_distance_action(guarded, status, rules, driver_id)
        guarded = self._guard_reposition_location_bounds_action(guarded, rules)
        guarded = self._guard_reposition_location_exclusion_action(guarded, rules)
        guarded = self._guard_reposition_daily_home_action(guarded, status, rules, decision_time_min)
        guarded = self._guard_take_order_distance_action(guarded, rules, driver_id, reachable_items)
        guarded = self._guard_take_order_action(guarded, reachable_items)
        self._remember_city_goal_action(driver_id, rules, guarded, reachable_items, decision_time_min)
        return guarded

    def build_market_sample(self, current_minute: int, items: list[dict[str, Any]]) -> dict[str, Any]:
        nets = sorted((self._estimate_order_net_income(item) for item in items), reverse=True)
        profitable_count = sum(1 for net in nets if net > 0)
        top_nets = nets[:_MARKET_TOP_NET_COUNT]
        top_avg_net = sum(top_nets) / len(top_nets) if top_nets else 0.0
        score = top_avg_net + profitable_count * 10.0 if nets else -10000.0
        return {
            "minute": current_minute,
            "day": current_minute // 1440,
            "reachable_count": len(items),
            "profitable_count": profitable_count,
            "best_net": nets[0] if nets else 0.0,
            "top_avg_net": top_avg_net,
            "score": score,
        }

    def remember_market_sample(self, driver_id: str, sample: dict[str, Any]) -> None:
        history = self._market_history_by_driver.setdefault(driver_id, [])
        history.append(sample)
        if len(history) > _MARKET_HISTORY_MAX_SAMPLES:
            del history[: len(history) - _MARKET_HISTORY_MAX_SAMPLES]

    def max_safe_idle_wait_minutes(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
        requested_minutes: int,
    ) -> int:
        requested = max(1, int(requested_minutes))
        takeover_minute = self._next_pre_query_takeover_minute(driver_id, status, rules, current_minute)
        if takeover_minute is None:
            return requested
        if takeover_minute <= current_minute:
            return 1
        return max(1, min(requested, takeover_minute - current_minute))

    def month_day_market_wait_minutes(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
        market_sample: dict[str, Any],
    ) -> int:
        day = current_minute // 1440
        if day >= _MONTHLY_REST_SCORING_DAYS:
            return 0
        month_rules = self._month_day_rules(rules)
        if not month_rules:
            return 0
        records = self._history_records(driver_id)
        for rule in month_rules:
            debt = self._month_day_debt(rule, records, current_minute)
            if int(debt["needed_days"]) <= 0 or not bool(debt["current_clean"]):
                continue
            wait_minutes = self._minutes_until_next_day(current_minute)
            if self._wait_crosses_pre_query_takeover(driver_id, status, rules, current_minute, wait_minutes):
                continue
            if day in self._month_fallback_rest_days(rule, records, current_minute, rules):
                return wait_minutes
            if self._is_low_market_for_driver(driver_id, market_sample):
                return wait_minutes
        return 0

    @staticmethod
    def unknown_preference_contents(rules: list[dict[str, Any]]) -> list[str]:
        return [
            str(rule.get("source_content", "") or "")
            for rule in rules
            if rule.get("rule_type") == "unknown" and str(rule.get("source_content", "") or "").strip()
        ]

    @staticmethod
    def rule_type_counts(rules: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in rules:
            rule_type = str(rule.get("rule_type", "unknown") or "unknown")
            counts[rule_type] = counts.get(rule_type, 0) + 1
        return counts

    def _scheduled_event_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        for rule in self._rules(rules, "scheduled_event"):
            params = self._params(rule)
            if params.get("mode") == "stops":
                action = self._scheduled_stops_action(driver_id, status, rules, params, current_minute)
            else:
                action = self._scheduled_pickup_home_action(driver_id, status, rules, params, current_minute)
            if action is not None:
                return action
        return None

    def _next_pre_query_takeover_minute(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> int | None:
        candidates: list[int] = []
        records = self._history_records(driver_id)
        self._collect_must_take_takeover_minutes(candidates, records, rules, current_minute)
        self._collect_scheduled_event_takeover_minutes(candidates, records, status, rules, current_minute)
        self._collect_daily_home_takeover_minutes(candidates, status, rules, current_minute)
        self._collect_order_cadence_takeover_minutes(candidates, records, rules, current_minute)
        future = [minute for minute in candidates if minute >= current_minute]
        return min(future) if future else None

    def _collect_must_take_takeover_minutes(
        self,
        candidates: list[int],
        records: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> None:
        rule = self._first_rule(rules, "must_take_cargo")
        if rule is None:
            return
        params = self._params(rule)
        cargo_id = str(params.get("cargo_id", ""))
        if cargo_id and self._has_taken_cargo(records, cargo_id):
            return
        active_start = self._safe_int(params.get("active_start_minute"))
        active_end = self._safe_int(params.get("active_end_minute"))
        if active_start is None or active_end is None or current_minute > active_end:
            return
        if current_minute >= active_start:
            candidates.append(current_minute)
            return
        prepare_start = active_start - 12 * 60
        candidates.append(max(current_minute, prepare_start))

    def _collect_scheduled_event_takeover_minutes(
        self,
        candidates: list[int],
        records: list[dict[str, Any]],
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> None:
        for rule in self._rules(rules, "scheduled_event"):
            params = self._params(rule)
            if params.get("mode") == "stops":
                active_start = self._safe_int(params.get("active_start_minute"))
                active_end = self._safe_int(params.get("active_end_minute"))
                if active_start is None or active_end is None or current_minute > active_end:
                    continue
                stop_index = self._next_pending_scheduled_stop_index(records, rules, params)
                if stop_index is None:
                    continue
                reservation_start = self._scheduled_stop_reservation_start(status, stop_index, params, rules)
                candidates.append(max(current_minute, reservation_start))
            else:
                event_start = self._safe_int(params.get("event_start_minute"))
                stay_until = self._safe_int(params.get("stay_until_minute"))
                if event_start is None or stay_until is None or current_minute >= stay_until:
                    continue
                reservation_start = self._scheduled_pickup_home_reservation_start(status, params, rules)
                candidates.append(max(current_minute, reservation_start))

    def _collect_daily_home_takeover_minutes(
        self,
        candidates: list[int],
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> None:
        params = self._first_rule_params(rules, "daily_home")
        if params is None:
            return
        lat = self._safe_float(status.get("current_lat"))
        lng = self._safe_float(status.get("current_lng"))
        home_lat = self._safe_float(params.get("home_lat"))
        home_lng = self._safe_float(params.get("home_lng"))
        deadline_minute = self._safe_int(params.get("deadline_minute"))
        if None in (lat, lng, home_lat, home_lng, deadline_minute):
            return
        radius_km = float(params.get("radius_km", 1.0) or 1.0)
        if self._near_point(float(lat), float(lng), float(home_lat), float(home_lng), radius_km):
            return
        day_start = (current_minute // 1440) * 1440
        deadline = day_start + int(deadline_minute)
        if current_minute < deadline:
            travel_minutes = self._pickup_minutes(self._haversine_km(float(lat), float(lng), float(home_lat), float(home_lng)))
            candidates.append(max(current_minute, deadline - 10 - travel_minutes))
        quiet_start = self._safe_int(params.get("quiet_start_minute"))
        quiet_end = self._safe_int(params.get("quiet_end_minute"))
        if quiet_start is not None and quiet_end is not None:
            quiet_start_abs = day_start + quiet_start
            if quiet_start_abs < current_minute:
                quiet_start_abs += 1440
            candidates.append(quiet_start_abs)

    def _collect_order_cadence_takeover_minutes(
        self,
        candidates: list[int],
        records: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> None:
        cadence = self._order_cadence_rule(rules)
        first_order_before = cadence.get("first_order_before_minute")
        if first_order_before is None:
            return
        day = current_minute // 1440
        if self._accepted_order_count_by_action_day(records, day) > 0:
            return
        first_order_minute = day * 1440 + int(first_order_before)
        if current_minute < first_order_minute:
            candidates.append(first_order_minute)

    def _scheduled_pickup_home_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        params: dict[str, Any],
        current_minute: int,
    ) -> dict[str, Any] | None:
        if current_minute >= int(params["stay_until_minute"]):
            return None

        records = self._history_records(driver_id)
        pickup_done = self._event_pickup_done(records, params)
        lat = float(status.get("current_lat", 0.0))
        lng = float(status.get("current_lng", 0.0))
        pickup_lat = float(params["pickup_lat"])
        pickup_lng = float(params["pickup_lng"])
        home_lat = float(params["home_lat"])
        home_lng = float(params["home_lng"])
        radius_km = float(params["radius_km"])
        near_pickup = self._near_point(lat, lng, pickup_lat, pickup_lng, radius_km)
        near_home = self._near_point(lat, lng, home_lat, home_lng, radius_km)

        if current_minute < int(params["event_start_minute"]):
            if current_minute < int(params["event_start_minute"]) - 24 * 60:
                return None
            if near_pickup:
                return {
                    "action": "wait",
                    "params": {"duration_minutes": max(1, int(params["event_start_minute"]) - current_minute)},
                }
            return self._reposition_or_quiet_wait(rules, current_minute, lat, lng, pickup_lat, pickup_lng)

        if not pickup_done:
            if near_pickup:
                return {"action": "wait", "params": {"duration_minutes": int(params["pickup_stay_minutes"])}}
            return self._reposition_or_quiet_wait(rules, current_minute, lat, lng, pickup_lat, pickup_lng)

        if current_minute < int(params["stay_until_minute"]):
            if near_home:
                return {"action": "wait", "params": {"duration_minutes": max(1, int(params["stay_until_minute"]) - current_minute)}}
            return self._reposition_or_quiet_wait(rules, current_minute, lat, lng, home_lat, home_lng)
        return None

    def _scheduled_stops_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        params: dict[str, Any],
        current_minute: int,
    ) -> dict[str, Any] | None:
        active_start = int(params["active_start_minute"])
        active_end = int(params["active_end_minute"])
        if current_minute > active_end:
            return None
        records = self._history_records(driver_id)
        stop_index = self._next_pending_scheduled_stop_index(records, rules, params)
        if stop_index is None:
            return None
        stop = params["stops"][stop_index]
        point = self._resolve_stop_point(stop, rules)
        if point is None:
            self._logger.warning("scheduled stop unresolved point stop=%s", stop)
            return self._unresolved_scheduled_stop_action(stop_index, params, current_minute, rules)

        lat = float(status.get("current_lat", 0.0))
        lng = float(status.get("current_lng", 0.0))
        target_lat, target_lng = point
        radius_km = float(stop.get("radius_km", 1.0) or 1.0)
        near_target = self._near_point(lat, lng, target_lat, target_lng, radius_km)

        arrive_after = self._safe_int(stop.get("arrive_after_minute"))
        if near_target and arrive_after is not None and current_minute < arrive_after:
            return {"action": "wait", "params": {"duration_minutes": max(1, arrive_after - current_minute)}}

        stay_until = self._safe_int(stop.get("stay_until_minute"))
        if near_target and stay_until is not None and current_minute < stay_until:
            return {"action": "wait", "params": {"duration_minutes": max(1, stay_until - current_minute)}}

        stay_minutes = self._safe_int(stop.get("stay_minutes"))
        if near_target and stay_minutes is not None and not self._scheduled_stop_done(records, rules, params, stop):
            return {"action": "wait", "params": {"duration_minutes": max(1, stay_minutes)}}

        if near_target and current_minute < active_start:
            return {"action": "wait", "params": {"duration_minutes": max(1, active_start - current_minute)}}

        if near_target:
            return None

        travel_minutes = self._pickup_minutes(self._haversine_km(lat, lng, target_lat, target_lng))
        deadline = self._scheduled_stop_effective_deadline(stop_index, params, rules)
        prepare_start = self._scheduled_stop_reservation_start(status, stop_index, params, rules)
        should_prepare = (
            current_minute >= prepare_start
            or current_minute >= active_start
            or current_minute + travel_minutes >= deadline - 60
        )
        if not should_prepare:
            return None
        if current_minute + travel_minutes > deadline:
            self._logger.info("scheduled stop deadline already infeasible stop=%s deadline=%s", stop, deadline)
            return None
        return self._reposition_or_quiet_wait(rules, current_minute, lat, lng, target_lat, target_lng)

    def _must_take_cargo_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        rule = self._first_rule(rules, "must_take_cargo")
        if rule is None:
            return None
        params = self._params(rule)
        records = self._history_records(driver_id)
        cargo_id = str(params["cargo_id"])
        if self._has_taken_cargo(records, cargo_id):
            return None
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        if current_minute > int(params["active_end_minute"]):
            return None

        lat = float(status.get("current_lat", 0.0))
        lng = float(status.get("current_lng", 0.0))
        pickup_lat = float(params["pickup_lat"])
        pickup_lng = float(params["pickup_lng"])
        if int(params["active_start_minute"]) <= current_minute <= int(params["active_end_minute"]):
            self._logger.info("priority must-take cargo_id=%s", cargo_id)
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}

        if current_minute < int(params["active_start_minute"]):
            home_rule = self._first_rule(rules, "daily_home")
            if home_rule is not None and self._daily_home_quiet_end(self._params(home_rule), current_minute) is not None:
                return None
            travel_minutes = self._pickup_minutes(self._haversine_km(lat, lng, pickup_lat, pickup_lng))
            prepare_start = int(params["active_start_minute"]) - 12 * 60
            if current_minute < prepare_start:
                return None
            if self._near_point(lat, lng, pickup_lat, pickup_lng, 1.0):
                return {"action": "wait", "params": {"duration_minutes": max(1, int(params["active_start_minute"]) - current_minute)}}
            if current_minute + travel_minutes <= int(params["active_end_minute"]):
                return {"action": "reposition", "params": {"latitude": pickup_lat, "longitude": pickup_lng}}
        return None

    def _month_day_fallback_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        day = current_minute // 1440
        if day >= _MONTHLY_REST_SCORING_DAYS:
            return None
        records = self._history_records(driver_id)
        for rule in self._month_day_rules(rules):
            debt = self._month_day_debt(rule, records, current_minute)
            if int(debt["needed_days"]) <= 0 or not bool(debt["current_clean"]):
                continue
            if day in self._month_fallback_rest_days(rule, records, current_minute, rules):
                wait_minutes = self._minutes_until_next_day(current_minute)
                if self._wait_crosses_pre_query_takeover(driver_id, status, rules, current_minute, wait_minutes):
                    continue
                return {"action": "wait", "params": {"duration_minutes": wait_minutes}}
        return None

    def _daily_home_action(self, status: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any] | None:
        rule = self._first_rule(rules, "daily_home")
        if rule is None:
            return None
        params = self._params(rule)
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        lat = float(status.get("current_lat", 0.0))
        lng = float(status.get("current_lng", 0.0))
        home_lat = float(params["home_lat"])
        home_lng = float(params["home_lng"])
        at_home = self._near_point(lat, lng, home_lat, home_lng, float(params["radius_km"]))

        quiet_end = self._daily_home_quiet_end(params, current_minute)
        if quiet_end is not None:
            if at_home:
                return {"action": "wait", "params": {"duration_minutes": max(1, quiet_end - current_minute)}}
            return {"action": "reposition", "params": {"latitude": home_lat, "longitude": home_lng}}

        deadline = (current_minute // 1440) * 1440 + int(params["deadline_minute"])
        if current_minute < deadline and not at_home:
            travel_minutes = self._pickup_minutes(self._haversine_km(lat, lng, home_lat, home_lng))
            if current_minute + travel_minutes >= deadline - 10:
                return {"action": "reposition", "params": {"latitude": home_lat, "longitude": home_lng}}
        return None

    def _monthly_visit_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        rule = self._first_rule(rules, "point_visit")
        if rule is None:
            return None
        event_rule = self._first_rule(rules, "scheduled_event")
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        if event_rule is not None:
            event_params = self._params(event_rule)
            if event_params.get("mode") == "stops":
                active_start = self._safe_int(event_params.get("active_start_minute"))
                active_end = self._safe_int(event_params.get("active_end_minute"))
                if active_start is not None and active_end is not None and active_start <= current_minute < active_end:
                    return None
            else:
                event_start = self._safe_int(event_params.get("event_start_minute"))
                stay_until = self._safe_int(event_params.get("stay_until_minute"))
                if event_start is not None and stay_until is not None and event_start <= current_minute < stay_until:
                    return None
        day = current_minute // 1440
        if day >= _MONTHLY_REST_SCORING_DAYS:
            return None

        params = self._params(rule)
        records = self._history_records(driver_id)
        visit_days = self._visited_days(records, float(params["target_lat"]), float(params["target_lng"]), float(params["radius_km"]))
        if len(visit_days) >= int(params["required_days"]) or day in visit_days:
            return None
        needed = int(params["required_days"]) - len(visit_days)
        remaining_days = _MONTHLY_REST_SCORING_DAYS - day
        lat = float(status.get("current_lat", 0.0))
        lng = float(status.get("current_lng", 0.0))
        target_lat = float(params["target_lat"])
        target_lng = float(params["target_lng"])
        distance_km = self._haversine_km(lat, lng, target_lat, target_lng)
        if distance_km <= float(params["radius_km"]):
            return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}
        if distance_km <= 30.0 or remaining_days <= needed:
            return {"action": "reposition", "params": {"latitude": target_lat, "longitude": target_lng}}
        return None

    def _city_goal_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        current_day = current_minute // 1440
        if current_day >= _SIMULATION_DAYS:
            return None
        if self._quiet_window_wait_minutes(rules, current_minute) > 0:
            return None
        for rule in self._rules(rules, "cargo_city_day_goal"):
            params = self._params(rule)
            target_lat = self._safe_float(params.get("target_lat"))
            target_lng = self._safe_float(params.get("target_lng"))
            if target_lat is None or target_lng is None:
                continue
            key = self._city_goal_key(params)
            done_days = self._city_goal_days_by_driver.get(driver_id, {}).get(key, set())
            required_days = int(params.get("required_days", 0) or 0)
            needed_days = required_days - len(done_days)
            if needed_days <= 0 or current_day in done_days:
                continue
            remaining_days = max(1, _SIMULATION_DAYS - current_day)
            expected_days = required_days * min(_SIMULATION_DAYS, current_day + 1) / _SIMULATION_DAYS
            behind = current_day >= 7 and len(done_days) + 0.5 < expected_days
            urgent = remaining_days <= needed_days * 5
            if not (behind or urgent):
                continue
            lat = float(status.get("current_lat", 0.0))
            lng = float(status.get("current_lng", 0.0))
            radius_km = float(params.get("radius_km", 1.0) or 1.0)
            if self._near_point(lat, lng, float(target_lat), float(target_lng), radius_km):
                return None
            self._logger.info("city goal reposition driver_id=%s goal=%s needed_days=%s", driver_id, key, needed_days)
            return self._reposition_or_quiet_wait(rules, current_minute, lat, lng, float(target_lat), float(target_lng))
        return None

    def _filter_rejected_cargo_items(self, items: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rejected: set[str] = set()
        for rule in self._rules(rules, "reject_cargo_category"):
            params = self._params(rule)
            if bool(params.get("hard", True)):
                rejected.update(str(item).strip() for item in params.get("categories", []) if str(item).strip())
        if not rejected:
            return items
        filtered: list[dict[str, Any]] = []
        for item in items:
            cargo = item.get("cargo", {})
            cargo_name = str(cargo.get("cargo_name", "") if isinstance(cargo, dict) else "").strip()
            if cargo_name in rejected:
                continue
            filtered.append(item)
        return filtered

    def _filter_city_rule_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        decision_time_min: int,
        *,
        hard: bool,
    ) -> list[dict[str, Any]]:
        city_rules = [
            self._params(rule)
            for rule in self._rules(rules, "cargo_city_filter")
            if bool(self._params(rule).get("hard", True)) == hard
        ]
        if not city_rules or not items:
            return items
        kept = [
            item
            for item in items
            if not any(self._cargo_city_filter_matches(item, params, decision_time_min) for params in city_rules)
        ]
        removed = len(items) - len(kept)
        if not hard and not kept:
            return items
        if removed:
            self._logger.info("filtered cargo city preference count=%s hard=%s", removed, hard)
        return kept

    def _filter_city_goal_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        driver_id: str,
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        if not items:
            return items
        current_day = decision_time_min // 1440
        for rule in self._rules(rules, "cargo_city_day_goal"):
            params = self._params(rule)
            key = self._city_goal_key(params)
            done_days = self._city_goal_days_by_driver.get(driver_id, {}).get(key, set())
            required_days = int(params.get("required_days", 0) or 0)
            needed_days = required_days - len(done_days)
            if needed_days <= 0 or current_day in done_days:
                continue
            matching = [item for item in items if self._cargo_city_matches(item, params)]
            if not matching:
                continue
            remaining_days = max(1, _SIMULATION_DAYS - current_day)
            expected_days = required_days * min(_SIMULATION_DAYS, current_day + 1) / _SIMULATION_DAYS
            should_catch_up = len(done_days) + 0.5 < expected_days and current_day >= 7
            if should_catch_up or remaining_days <= needed_days * 3:
                removed = len(items) - len(matching)
                if removed:
                    self._logger.info("preferred cargo city goal count=%s goal=%s", removed, key)
                return matching
        return items

    def _filter_soft_rejected_cargo_items(self, items: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        soft_rejected: set[str] = set()
        for rule in self._rules(rules, "reject_cargo_category"):
            params = self._params(rule)
            if not bool(params.get("hard", True)):
                soft_rejected.update(str(item).strip() for item in params.get("categories", []) if str(item).strip())
        if not soft_rejected:
            return items
        preferred = [item for item in items if self._cargo_name(item) not in soft_rejected]
        if not preferred:
            return items
        removed = len(items) - len(preferred)
        if removed:
            self._logger.info("filtered soft cargo preference count=%s categories=%s", removed, sorted(soft_rejected))
        return preferred

    def _filter_daily_home_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        driver_id: str,
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        rule = self._first_rule(rules, "daily_home")
        if rule is None or not items:
            return items
        params = self._params(rule)
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            finish_minutes = self._safe_float(item.get("finish_minutes"))
            if finish_minutes is None:
                filtered.append(item)
                continue
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            cargo_id = str(cargo.get("cargo_id", "")).strip()
            if self.is_active_must_take_cargo(driver_id, rules, cargo_id, decision_time_min):
                filtered.append(item)
                continue
            end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
            end_lat = self._safe_float(end.get("lat"))
            end_lng = self._safe_float(end.get("lng"))
            if end_lat is None or end_lng is None or self._daily_home_item_conflicts(
                decision_time_min,
                int(finish_minutes),
                float(end_lat),
                float(end_lng),
                params,
            ):
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered daily-home cargo count=%s", removed)
        return filtered

    def is_active_must_take_cargo(
        self,
        driver_id: str,
        rules: list[dict[str, Any]],
        cargo_id: str,
        current_minute: int,
    ) -> bool:
        cargo_id = str(cargo_id).strip()
        if not cargo_id:
            return False
        rule = self._first_rule(rules, "must_take_cargo")
        if rule is None:
            return False
        params = self._params(rule)
        if cargo_id != str(params.get("cargo_id", "")).strip():
            return False
        if self._has_taken_cargo(self._history_records(driver_id), cargo_id):
            return False
        active_start = self._safe_int(params.get("active_start_minute"))
        active_end = self._safe_int(params.get("active_end_minute"))
        if active_start is None or active_end is None:
            return False
        return active_start <= current_minute <= active_end

    def _filter_location_bounds_items(self, items: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rule = self._first_rule(rules, "location_bounds")
        if rule is None or not items:
            return items
        bounds = self._params(rule)
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
            end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
            points = [
                (self._safe_float(start.get("lat")), self._safe_float(start.get("lng"))),
                (self._safe_float(end.get("lat")), self._safe_float(end.get("lng"))),
            ]
            if any(lat is None or lng is None or not self._in_location_bounds(float(lat), float(lng), bounds) for lat, lng in points):
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered location-bounds cargo count=%s bounds=%s", removed, bounds)
        return filtered

    def _filter_location_exclusion_items(self, items: list[dict[str, Any]], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        circle = self._first_rule_params(rules, "location_exclusion_circle")
        if circle is None or not items:
            return items
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
            end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
            points = [
                (self._safe_float(start.get("lat")), self._safe_float(start.get("lng"))),
                (self._safe_float(end.get("lat")), self._safe_float(end.get("lng"))),
            ]
            if any(lat is None or lng is None or self._inside_exclusion_circle(float(lat), float(lng), circle) for lat, lng in points):
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered location-exclusion cargo count=%s circle=%s", removed, circle)
        return filtered

    def _filter_distance_rule_items(self, items: list[dict[str, Any]], rules: list[dict[str, Any]], driver_id: str) -> list[dict[str, Any]]:
        distance_rules = self._distance_rules(rules)
        if not distance_rules or not items:
            return items

        used_deadhead_km = 0.0
        if distance_rules.get("max_month_deadhead_km") is not None:
            used_deadhead_km = self._month_deadhead_km(self._history_records(driver_id))

        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            pickup_km = self._safe_float(item.get("distance_km"))
            haul_km = self._cargo_haul_distance_km(cargo)

            max_haul = distance_rules.get("max_haul_km")
            if max_haul is not None and haul_km > float(max_haul):
                removed += 1
                continue

            max_pickup = distance_rules.get("max_pickup_km")
            if max_pickup is not None and pickup_km is not None and pickup_km > float(max_pickup):
                removed += 1
                continue

            max_month_deadhead = distance_rules.get("max_month_deadhead_km")
            if max_month_deadhead is not None and pickup_km is not None:
                cap_km = float(max_month_deadhead)
                if used_deadhead_km + pickup_km > cap_km:
                    removed += 1
                    continue

            filtered.append(item)
        if removed:
            self._logger.info("filtered distance-rule cargo count=%s rules=%s", removed, distance_rules)
        return filtered

    def _filter_scheduled_event_conflict_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        driver_id: str,
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        if not items:
            return items
        records = self._history_records(driver_id)
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            if self._item_conflicts_scheduled_events(item, rules, records, decision_time_min):
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered scheduled-event conflict cargo count=%s", removed)
        return filtered

    def _item_conflicts_scheduled_events(
        self,
        item: dict[str, Any],
        rules: list[dict[str, Any]],
        records: list[dict[str, Any]],
        decision_time_min: int,
    ) -> bool:
        finish_minutes = self._safe_int(item.get("finish_minutes"))
        if finish_minutes is None:
            return False
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        end_lat = self._safe_float(end.get("lat"))
        end_lng = self._safe_float(end.get("lng"))
        if end_lat is None or end_lng is None:
            return False
        for rule in self._rules(rules, "scheduled_event"):
            params = self._params(rule)
            if params.get("mode") != "stops":
                continue
            active_start = int(params["active_start_minute"])
            active_end = int(params["active_end_minute"])
            if decision_time_min > active_end or decision_time_min < active_start - 24 * 60:
                continue
            stop_index = self._next_pending_scheduled_stop_index(records, rules, params)
            if stop_index is None:
                continue
            stop = params["stops"][stop_index]
            point = self._resolve_stop_point(stop, rules)
            if point is None:
                if self._unresolved_scheduled_stop_blocks_activity(stop_index, params, decision_time_min, int(finish_minutes), rules):
                    return True
                continue
            target_lat, target_lng = point
            travel_minutes = self._pickup_minutes(self._haversine_km(float(end_lat), float(end_lng), target_lat, target_lng))
            deadline = self._scheduled_stop_effective_deadline(stop_index, params, rules)
            latest_departure = self._latest_departure_without_quiet_conflict(rules, deadline, travel_minutes)
            if finish_minutes > latest_departure:
                return True
        return False

    def _daily_rest_wait_minutes(self, driver_id: str, status: dict[str, Any], rules: list[dict[str, Any]]) -> int:
        rule = self._first_rule(rules, "daily_rest")
        if rule is None:
            return 0
        params = self._params(rule)
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        day = current_minute // 1440
        if params["applies_on"] == "weekday" and (_SIMULATION_EPOCH.weekday() + day) % 7 >= 5:
            return 0

        records = self._history_records(driver_id)
        longest_wait = self._longest_wait_minutes_for_day(records, day)
        rest_minutes = int(params["minutes"])
        if int(params.get("required_count", 1)) <= 0 or longest_wait >= rest_minutes:
            return 0
        return rest_minutes

    def _filter_rest_conflict_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        rule = self._first_rule(rules, "daily_rest")
        if rule is None or not items:
            return items
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            penalty = self._estimate_rest_conflict_penalty(item, rule, decision_time_min)
            if penalty > 0:
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered rest-conflict cargo count=%s", removed)
        return filtered

    def _calendar_preference_wait_minutes(self, driver_id: str, status: dict[str, Any], rules: list[dict[str, Any]]) -> int:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        day = current_minute // 1440
        if day >= _SIMULATION_DAYS:
            return 0
        cadence = self._order_cadence_rule(rules)
        if cadence:
            return self._order_cadence_wait_minutes(cadence, self._history_records(driver_id), current_minute)
        return 0

    def _filter_month_day_conflict_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        driver_id: str,
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        month_rules = self._month_day_rules(rules)
        if not month_rules or not items:
            return items

        records = self._history_records(driver_id)
        protected: list[tuple[int, str]] = []
        for rule in month_rules:
            for day in self._month_fallback_rest_days(rule, records, decision_time_min, rules):
                protected.append((day, str(rule["mode"])))
        if not protected:
            return items

        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            finish_minutes = self._safe_float(item.get("finish_minutes"))
            if finish_minutes is None:
                filtered.append(item)
                continue
            if self._month_day_item_conflicts(decision_time_min, int(finish_minutes), protected):
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered month-day rest conflict cargo count=%s", removed)
        return filtered

    def _quiet_window_wait_minutes(self, rules: list[dict[str, Any]], current_minute: int) -> int:
        wait_until = current_minute
        for rule in self._quiet_window_rules(rules):
            for day in (current_minute // 1440 - 1, current_minute // 1440):
                if day < 0:
                    continue
                start, end = self._quiet_window_interval(rule, day)
                if start <= current_minute < end:
                    wait_until = max(wait_until, end)
        return max(0, wait_until - current_minute)

    def _filter_quiet_window_conflict_items(
        self,
        items: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        quiet_rules = self._quiet_window_rules(rules)
        if not quiet_rules or not items:
            return items
        filtered: list[dict[str, Any]] = []
        removed = 0
        for item in items:
            finish_minutes = self._safe_float(item.get("finish_minutes"))
            if finish_minutes is None:
                filtered.append(item)
                continue
            penalty = self._estimate_quiet_window_penalty(decision_time_min, int(finish_minutes), quiet_rules)
            if penalty > 0:
                removed += 1
                continue
            filtered.append(item)
        if removed:
            self._logger.info("filtered quiet-window conflict cargo count=%s", removed)
        return filtered

    def _guard_reposition_quiet_window_action(
        self,
        action: dict[str, Any],
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        decision_time_min: int,
    ) -> dict[str, Any]:
        if action.get("action") != "reposition":
            return action
        quiet_rules = self._quiet_window_rules(rules)
        if not quiet_rules:
            return action
        params = action.get("params") or {}
        distance_km = self._haversine_km(
            float(status.get("current_lat", 0.0)),
            float(status.get("current_lng", 0.0)),
            float(params.get("latitude")),
            float(params.get("longitude")),
        )
        end_minute = decision_time_min + self._pickup_minutes(distance_km)
        if self._estimate_quiet_window_penalty(decision_time_min, end_minute, quiet_rules) <= 0:
            return action
        wait_minutes = self._quiet_window_wait_minutes(rules, decision_time_min)
        if wait_minutes <= 0:
            wait_minutes = self._minutes_until_next_quiet_window_start(quiet_rules, decision_time_min)
        return {"action": "wait", "params": {"duration_minutes": max(1, wait_minutes)}}

    def _guard_reposition_distance_action(
        self,
        action: dict[str, Any],
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        driver_id: str,
    ) -> dict[str, Any]:
        if action.get("action") != "reposition":
            return action
        distance_rules = self._distance_rules(rules)
        max_month_deadhead = distance_rules.get("max_month_deadhead_km")
        if max_month_deadhead is None:
            return action
        params = action.get("params") or {}
        lat = self._safe_float(params.get("latitude"))
        lng = self._safe_float(params.get("longitude"))
        cur_lat = self._safe_float(status.get("current_lat"))
        cur_lng = self._safe_float(status.get("current_lng"))
        if None in (lat, lng, cur_lat, cur_lng):
            return action
        distance_km = self._haversine_km(float(cur_lat), float(cur_lng), float(lat), float(lng))
        used_deadhead_km = self._month_deadhead_km(self._history_records(driver_id))
        if used_deadhead_km + distance_km <= float(max_month_deadhead):
            return action
        self._logger.info(
            "replace reposition by month deadhead cap driver_id=%s used=%.2f add=%.2f cap=%.2f",
            driver_id,
            used_deadhead_km,
            distance_km,
            float(max_month_deadhead),
        )
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

    def _guard_reposition_location_bounds_action(self, action: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
        if action.get("action") != "reposition":
            return action
        bounds = self._first_rule_params(rules, "location_bounds")
        if bounds is None:
            return action
        params = action.get("params") or {}
        lat = self._safe_float(params.get("latitude"))
        lng = self._safe_float(params.get("longitude"))
        if lat is None or lng is None:
            return action
        if self._in_location_bounds(float(lat), float(lng), bounds):
            return action
        self._logger.info("replace reposition by location bounds target=(%.5f,%.5f) bounds=%s", float(lat), float(lng), bounds)
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

    def _guard_reposition_location_exclusion_action(self, action: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
        if action.get("action") != "reposition":
            return action
        circle = self._first_rule_params(rules, "location_exclusion_circle")
        if circle is None:
            return action
        params = action.get("params") or {}
        lat = self._safe_float(params.get("latitude"))
        lng = self._safe_float(params.get("longitude"))
        if lat is None or lng is None:
            return action
        if not self._inside_exclusion_circle(float(lat), float(lng), circle):
            return action
        self._logger.info("replace reposition by location exclusion target=(%.5f,%.5f) circle=%s", float(lat), float(lng), circle)
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

    def _guard_reposition_daily_home_action(
        self,
        action: dict[str, Any],
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        decision_time_min: int,
    ) -> dict[str, Any]:
        if action.get("action") != "reposition":
            return action
        home = self._first_rule_params(rules, "daily_home")
        if home is None:
            return action
        params = action.get("params") or {}
        lat = self._safe_float(params.get("latitude"))
        lng = self._safe_float(params.get("longitude"))
        cur_lat = self._safe_float(status.get("current_lat"))
        cur_lng = self._safe_float(status.get("current_lng"))
        if None in (lat, lng, cur_lat, cur_lng):
            return action
        distance_km = self._haversine_km(float(cur_lat), float(cur_lng), float(lat), float(lng))
        end_minute = decision_time_min + self._pickup_minutes(distance_km)
        if self._daily_home_quiet_end(home, decision_time_min) is not None:
            return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}
        deadline = (decision_time_min // 1440) * 1440 + int(home["deadline_minute"])
        home_minutes = self._pickup_minutes(self._haversine_km(float(lat), float(lng), float(home["home_lat"]), float(home["home_lng"])))
        if decision_time_min < deadline and end_minute + home_minutes > deadline:
            return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}
        return action

    def _guard_take_order_distance_action(
        self,
        action: dict[str, Any],
        rules: list[dict[str, Any]],
        driver_id: str,
        reachable_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if action.get("action") != "take_order":
            return action
        max_month_deadhead = self._distance_rules(rules).get("max_month_deadhead_km")
        if max_month_deadhead is None:
            return action
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        if not cargo_id:
            return action
        item = self._find_reachable_item(cargo_id, reachable_items)
        if item is None:
            return action
        pickup_km = self._safe_float(item.get("distance_km"))
        if pickup_km is None:
            return action
        used_deadhead_km = self._month_deadhead_km(self._history_records(driver_id))
        if used_deadhead_km + pickup_km <= float(max_month_deadhead):
            return action
        self._logger.info(
            "replace take_order by month deadhead cap driver_id=%s cargo_id=%s used=%.2f add=%.2f cap=%.2f",
            driver_id,
            cargo_id,
            used_deadhead_km,
            pickup_km,
            float(max_month_deadhead),
        )
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

    def _guard_take_order_action(self, action: dict[str, Any], reachable_items: list[dict[str, Any]]) -> dict[str, Any]:
        if action.get("action") != "take_order":
            return action
        reachable_ids = {
            str((item.get("cargo") or {}).get("cargo_id", "")).strip()
            for item in reachable_items
            if isinstance(item.get("cargo"), dict)
        }
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        if cargo_id in reachable_ids:
            return action
        fallback_id = self._select_fallback_cargo_id(reachable_items)
        if fallback_id:
            self._logger.info("replace unreachable cargo_id=%s with fallback cargo_id=%s", cargo_id, fallback_id)
            return {"action": "take_order", "params": {"cargo_id": fallback_id}}
        self._logger.info("replace unreachable cargo_id=%s with wait", cargo_id)
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

    def _reposition_or_quiet_wait(
        self,
        rules: list[dict[str, Any]],
        current_minute: int,
        lat: float,
        lng: float,
        target_lat: float,
        target_lng: float,
    ) -> dict[str, Any]:
        quiet_rules = self._quiet_window_rules(rules)
        travel_minutes = self._pickup_minutes(self._haversine_km(lat, lng, target_lat, target_lng))
        if quiet_rules and self._estimate_quiet_window_penalty(current_minute, current_minute + travel_minutes, quiet_rules) > 0:
            wait_minutes = self._quiet_window_wait_minutes(rules, current_minute)
            if wait_minutes <= 0:
                wait_minutes = self._minutes_until_next_quiet_window_start(quiet_rules, current_minute)
            return {"action": "wait", "params": {"duration_minutes": max(1, wait_minutes)}}
        return {"action": "reposition", "params": {"latitude": target_lat, "longitude": target_lng}}

    def _event_pickup_done(self, records: list[dict[str, Any]], params: dict[str, Any]) -> bool:
        pickup_run = 0
        for ctx in self._step_contexts(records):
            if int(ctx["step_end"]) <= int(params["event_start_minute"]):
                continue
            at_pickup = self._near_point(
                float(ctx["after_lat"]),
                float(ctx["after_lng"]),
                float(params["pickup_lat"]),
                float(params["pickup_lng"]),
                float(params["radius_km"]),
            )
            if ctx["action_name"] == "wait" and at_pickup:
                pickup_run += int(ctx["action_end"]) - int(ctx["action_start"])
                if pickup_run >= int(params["pickup_stay_minutes"]):
                    return True
            elif not (
                self._near_point(
                    float(ctx["before_lat"]),
                    float(ctx["before_lng"]),
                    float(params["pickup_lat"]),
                    float(params["pickup_lng"]),
                    float(params["radius_km"]),
                )
                and at_pickup
            ):
                pickup_run = 0
        return False

    def _next_pending_scheduled_stop_index(
        self,
        records: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> int | None:
        for index, stop in enumerate(params.get("stops") or []):
            if not isinstance(stop, dict):
                continue
            if not self._scheduled_stop_done(records, rules, params, stop):
                return index
        return None

    def _scheduled_stop_done(
        self,
        records: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        params: dict[str, Any],
        stop: dict[str, Any],
    ) -> bool:
        point = self._resolve_stop_point(stop, rules)
        if point is None:
            return False
        target_lat, target_lng = point
        radius_km = float(stop.get("radius_km", 1.0) or 1.0)
        active_start = int(params["active_start_minute"])
        active_end = int(params["active_end_minute"])
        deadline = self._safe_int(stop.get("deadline_minute"))
        window_end = min(active_end, deadline) if deadline is not None else active_end
        stay_until = self._safe_int(stop.get("stay_until_minute"))
        stay_minutes = self._safe_int(stop.get("stay_minutes"))

        wait_run = 0
        for ctx in self._step_contexts(records):
            if int(ctx["action_end"]) < active_start or int(ctx["action_start"]) > active_end:
                continue
            at_target_after = self._near_point(
                float(ctx["after_lat"]),
                float(ctx["after_lng"]),
                target_lat,
                target_lng,
                radius_km,
            )
            if not at_target_after:
                wait_run = 0
                continue
            if stay_until is not None:
                if ctx["action_name"] == "wait" and int(ctx["action_end"]) >= stay_until:
                    return True
                continue
            if stay_minutes is not None:
                if ctx["action_name"] == "wait":
                    wait_run += max(0, int(ctx["action_end"]) - int(ctx["action_start"]))
                    if wait_run >= stay_minutes:
                        return True
                continue
            if int(ctx["action_end"]) <= window_end:
                return True
        return False

    def _scheduled_stop_arrival_deadline(self, stop: dict[str, Any], params: dict[str, Any]) -> int:
        deadline = self._safe_int(stop.get("deadline_minute"))
        if deadline is not None:
            return deadline
        stay_until = self._safe_int(stop.get("stay_until_minute"))
        if stay_until is not None:
            return stay_until
        stay_minutes = self._safe_int(stop.get("stay_minutes"))
        active_end = int(params["active_end_minute"])
        if stay_minutes is not None:
            return max(int(params["active_start_minute"]), active_end - stay_minutes)
        return active_end

    def _scheduled_stop_effective_deadline(
        self,
        stop_index: int,
        params: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> int:
        stops = params.get("stops") if isinstance(params.get("stops"), list) else []
        stop = stops[stop_index]
        deadline = self._scheduled_stop_arrival_deadline(stop, params)
        if stop_index + 1 >= len(stops):
            return deadline
        current_point = self._resolve_stop_point(stop, rules)
        next_stop = stops[stop_index + 1]
        next_point = self._resolve_stop_point(next_stop, rules)
        if current_point is None or next_point is None:
            return deadline
        next_deadline = self._scheduled_stop_arrival_deadline(next_stop, params)
        travel_minutes = self._pickup_minutes(self._haversine_km(current_point[0], current_point[1], next_point[0], next_point[1]))
        return min(deadline, max(int(params["active_start_minute"]), next_deadline - travel_minutes))

    def _unresolved_scheduled_stop_action(
        self,
        stop_index: int,
        params: dict[str, Any],
        current_minute: int,
        rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._unresolved_scheduled_stop_blocks_activity(
            stop_index,
            params,
            current_minute,
            current_minute + 1,
            rules,
        ):
            return None
        active_end = int(params["active_end_minute"])
        wait_minutes = min(_UNRESOLVED_EVENT_WAIT_MINUTES, max(1, active_end + 1 - current_minute))
        return {"action": "wait", "params": {"duration_minutes": wait_minutes}}

    def _unresolved_scheduled_stop_blocks_activity(
        self,
        stop_index: int,
        params: dict[str, Any],
        start_minute: int,
        finish_minute: int,
        rules: list[dict[str, Any]],
    ) -> bool:
        active_end = int(params["active_end_minute"])
        if start_minute > active_end:
            return False
        prepare_start = self._scheduled_prepare_start(stop_index, params, rules)
        return finish_minute > prepare_start

    def _scheduled_prepare_start(self, stop_index: int, params: dict[str, Any], rules: list[dict[str, Any]]) -> int:
        active_start = int(params["active_start_minute"])
        prepare_start = active_start - 12 * 60
        quiet_rules = self._quiet_window_rules(rules)
        if stop_index != 0 or not quiet_rules:
            return prepare_start
        stops = params.get("stops") if isinstance(params.get("stops"), list) else []
        has_morning_deadline = any(
            self._safe_int(stop.get("deadline_minute")) is not None
            and self._safe_int(stop.get("deadline_minute")) % 1440 <= 12 * 60
            for stop in stops
            if isinstance(stop, dict)
        )
        if not has_morning_deadline:
            return prepare_start
        quiet_end_minute = max(int(rule["end_minute"]) for rule in quiet_rules)
        return min(prepare_start, active_start - (1440 - quiet_end_minute))

    def _scheduled_stop_reservation_start(
        self,
        status: dict[str, Any],
        stop_index: int,
        params: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> int:
        fallback = self._scheduled_prepare_start(stop_index, params, rules)
        stops = params.get("stops") if isinstance(params.get("stops"), list) else []
        if stop_index >= len(stops):
            return fallback
        point = self._resolve_stop_point(stops[stop_index], rules)
        lat = self._safe_float(status.get("current_lat"))
        lng = self._safe_float(status.get("current_lng"))
        if point is None or lat is None or lng is None:
            return fallback
        deadline = self._scheduled_stop_effective_deadline(stop_index, params, rules)
        travel_minutes = self._pickup_minutes(self._haversine_km(float(lat), float(lng), point[0], point[1]))
        latest_departure = self._latest_departure_without_quiet_conflict(rules, deadline, travel_minutes)
        return min(fallback, latest_departure)

    def _scheduled_pickup_home_reservation_start(
        self,
        status: dict[str, Any],
        params: dict[str, Any],
        rules: list[dict[str, Any]],
    ) -> int:
        event_start = int(params["event_start_minute"])
        pickup_lat = self._safe_float(params.get("pickup_lat"))
        pickup_lng = self._safe_float(params.get("pickup_lng"))
        lat = self._safe_float(status.get("current_lat"))
        lng = self._safe_float(status.get("current_lng"))
        fallback = event_start - 24 * 60
        if None in (pickup_lat, pickup_lng, lat, lng):
            return fallback
        travel_minutes = self._pickup_minutes(
            self._haversine_km(float(lat), float(lng), float(pickup_lat), float(pickup_lng))
        )
        latest_departure = self._latest_departure_without_quiet_conflict(rules, event_start, travel_minutes)
        return min(fallback, latest_departure)

    def _latest_departure_without_quiet_conflict(
        self,
        rules: list[dict[str, Any]],
        deadline: int,
        travel_minutes: int,
    ) -> int:
        if travel_minutes <= 0:
            return deadline
        quiet_rules = self._quiet_window_rules(rules)
        latest_arrival = int(deadline)
        for _ in range(20):
            departure = latest_arrival - int(travel_minutes)
            conflict_starts: list[int] = []
            for rule in quiet_rules:
                start_day = max(0, departure // 1440 - 1)
                end_day = max(0, latest_arrival // 1440)
                for day in range(start_day, end_day + 1):
                    window_start, window_end = self._quiet_window_interval(rule, day)
                    if max(departure, window_start) < min(latest_arrival, window_end):
                        conflict_starts.append(window_start)
            if not conflict_starts:
                return departure
            latest_arrival = min(conflict_starts)
        return int(deadline) - int(travel_minutes)

    def _wait_crosses_scheduled_reservation(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
        wait_minutes: int,
    ) -> bool:
        candidates: list[int] = []
        records = self._history_records(driver_id)
        self._collect_scheduled_event_takeover_minutes(candidates, records, status, rules, current_minute)
        future = [minute for minute in candidates if minute >= current_minute]
        if not future:
            return False
        reservation_start = min(future)
        return current_minute < reservation_start < current_minute + max(1, int(wait_minutes))

    def _wait_crosses_pre_query_takeover(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        current_minute: int,
        wait_minutes: int,
    ) -> bool:
        takeover_minute = self._next_pre_query_takeover_minute(driver_id, status, rules, current_minute)
        if takeover_minute is None:
            return False
        if takeover_minute <= current_minute:
            return True
        return takeover_minute < current_minute + max(1, int(wait_minutes))

    def _resolve_stop_point(self, stop: dict[str, Any], rules: list[dict[str, Any]]) -> tuple[float, float] | None:
        lat = self._safe_float(stop.get("lat"))
        lng = self._safe_float(stop.get("lng"))
        if lat is not None and lng is not None:
            return float(lat), float(lng)
        keywords = [str(item).strip() for item in stop.get("place_keywords", []) if str(item).strip()]
        if not keywords:
            return None
        for rule in self._rules(rules, "cargo_city_day_goal"):
            params = self._params(rule)
            target_lat = self._safe_float(params.get("target_lat"))
            target_lng = self._safe_float(params.get("target_lng"))
            if target_lat is None or target_lng is None:
                continue
            city_keywords = [str(item).strip() for item in params.get("city_keywords", []) if str(item).strip()]
            if self._keyword_lists_overlap(keywords, city_keywords):
                return float(target_lat), float(target_lng)
        for rule in self._rules(rules, "point_visit"):
            params = self._params(rule)
            visit_keywords = [str(item).strip() for item in params.get("place_keywords", []) if str(item).strip()]
            if not visit_keywords or not self._keyword_lists_overlap(keywords, visit_keywords):
                continue
            target_lat = self._safe_float(params.get("target_lat"))
            target_lng = self._safe_float(params.get("target_lng"))
            if target_lat is not None and target_lng is not None:
                return float(target_lat), float(target_lng)
        return None

    @staticmethod
    def _keyword_lists_overlap(left: list[str], right: list[str]) -> bool:
        for a in left:
            for b in right:
                if a in b or b in a:
                    return True
        return False

    def _visited_days(self, records: list[dict[str, Any]], lat: float, lng: float, radius_km: float) -> set[int]:
        days: set[int] = set()
        for ctx in self._step_contexts(records):
            if self._near_point(float(ctx["after_lat"]), float(ctx["after_lng"]), lat, lng, radius_km):
                days.add(int(ctx["step_end"]) // 1440)
        return days

    def _has_taken_cargo(self, records: list[dict[str, Any]], cargo_id: str) -> bool:
        for record in records:
            if not isinstance(record, dict):
                continue
            action = record.get("action") if isinstance(record.get("action"), dict) else {}
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            if str(action.get("action", "")).strip().lower() != "take_order" or not bool(result.get("accepted", False)):
                continue
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            if str(params.get("cargo_id", "")).strip() == cargo_id:
                return True
        return False

    def _daily_home_quiet_end(self, params: dict[str, Any], current_minute: int) -> int | None:
        if "quiet_start_minute" not in params or "quiet_end_minute" not in params:
            return None
        minute_of_day = current_minute % 1440
        quiet_start = int(params["quiet_start_minute"])
        quiet_end = int(params["quiet_end_minute"])
        day = current_minute // 1440
        if quiet_end > quiet_start:
            if quiet_start <= minute_of_day < quiet_end:
                return day * 1440 + quiet_end
            return None
        if minute_of_day >= quiet_start:
            return (day + 1) * 1440 + quiet_end
        if minute_of_day < quiet_end:
            return day * 1440 + quiet_end
        return None

    def _daily_home_item_conflicts(
        self,
        start_minute: int,
        finish_minute: int,
        finish_lat: float,
        finish_lng: float,
        params: dict[str, Any],
    ) -> bool:
        if "quiet_start_minute" in params and "quiet_end_minute" in params:
            start_day = max(0, start_minute // 1440 - 1)
            end_day = max(start_day, finish_minute // 1440)
            quiet_start_minute = int(params["quiet_start_minute"])
            quiet_end_minute = int(params["quiet_end_minute"])
            for day in range(start_day, end_day + 1):
                quiet_start = day * 1440 + quiet_start_minute
                quiet_end = day * 1440 + quiet_end_minute
                if quiet_end <= quiet_start:
                    quiet_end += 1440
                if self._interval_overlap(start_minute, finish_minute, quiet_start, quiet_end):
                    return True

        deadline = (start_minute // 1440) * 1440 + int(params["deadline_minute"])
        if start_minute < deadline and finish_minute <= deadline:
            home_minutes = self._pickup_minutes(
                self._haversine_km(finish_lat, finish_lng, float(params["home_lat"]), float(params["home_lng"]))
            )
            return finish_minute + home_minutes > deadline
        return False

    @staticmethod
    def _in_location_bounds(lat: float, lng: float, bounds: dict[str, Any]) -> bool:
        return (
            float(bounds["lat_min"]) <= lat <= float(bounds["lat_max"])
            and float(bounds["lng_min"]) <= lng <= float(bounds["lng_max"])
        )

    def _inside_exclusion_circle(self, lat: float, lng: float, circle: dict[str, Any]) -> bool:
        return (
            self._haversine_km(lat, lng, float(circle["center_lat"]), float(circle["center_lng"]))
            <= float(circle["radius_km"])
        )

    def _month_deadhead_km(self, records: list[dict[str, Any]]) -> float:
        total = 0.0
        for record in records:
            if not isinstance(record, dict):
                continue
            action = record.get("action") if isinstance(record.get("action"), dict) else {}
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            action_name = str(action.get("action", "")).strip().lower()
            if action_name == "reposition":
                total += self._safe_float(result.get("distance_km")) or 0.0
            elif action_name == "take_order" and bool(result.get("accepted", False)):
                total += self._safe_float(result.get("pickup_deadhead_km")) or 0.0
        return total

    @staticmethod
    def _longest_wait_minutes_for_day(records: Any, day: int) -> int:
        if not isinstance(records, list):
            return 0
        day_start = day * 1440
        day_end = day_start + 1440
        intervals: list[tuple[int, int]] = []
        prev_end = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            try:
                step_end = int(result.get("simulation_progress_minutes", -1))
                action_exec_cost = int(record.get("action_exec_cost_minutes", -1))
            except (TypeError, ValueError):
                continue
            step_start = prev_end
            prev_end = step_end
            action = record.get("action") if isinstance(record.get("action"), dict) else {}
            if str(action.get("action", "")).strip().lower() != "wait" or action_exec_cost <= 0:
                continue
            start = max(step_start, day_start)
            end = min(step_end, day_end)
            if end > start:
                intervals.append((start, end))
        if not intervals:
            return 0

        longest = 0
        cur_start, cur_end = sorted(intervals)[0]
        for start, end in sorted(intervals)[1:]:
            if start > cur_end:
                longest = max(longest, cur_end - cur_start)
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        return max(longest, cur_end - cur_start)

    def _month_day_debt(self, rule: dict[str, Any], records: list[dict[str, Any]], current_minute: int) -> dict[str, Any]:
        day = current_minute // 1440
        satisfied_past = 0
        for past_day in range(min(day, _MONTHLY_REST_SCORING_DAYS)):
            if self._month_day_rule_satisfied(records, past_day, rule["mode"]):
                satisfied_past += 1
        needed = max(0, int(rule["required_days"]) - satisfied_past)
        current_clean = day < _MONTHLY_REST_SCORING_DAYS and self._month_day_rule_satisfied(records, day, rule["mode"])
        return {
            "required_days": int(rule["required_days"]),
            "satisfied_days": satisfied_past,
            "needed_days": needed,
            "current_clean": current_clean,
        }

    def _month_fallback_rest_days(
        self,
        rule: dict[str, Any],
        records: list[dict[str, Any]],
        current_minute: int,
        rules: list[dict[str, Any]] | None = None,
    ) -> set[int]:
        day = current_minute // 1440
        debt = self._month_day_debt(rule, records, current_minute)
        needed = int(debt["needed_days"])
        if needed <= 0 or day >= _MONTHLY_REST_SCORING_DAYS:
            return set()
        blocked = self._month_rest_blocked_days(rules or []) if str(rule.get("mode")) == "no_active" else set()
        start_day = day if bool(debt["current_clean"]) else day + 1
        future_candidates = [
            candidate for candidate in range(start_day, _MONTHLY_REST_SCORING_DAYS) if candidate not in blocked
        ]
        if len(future_candidates) <= needed:
            return set(future_candidates)

        all_candidates = [candidate for candidate in range(_MONTHLY_REST_SCORING_DAYS) if candidate not in blocked]
        planned_days = self._select_month_rest_plan_days(all_candidates, int(rule["required_days"]))
        selected = [candidate for candidate in planned_days if candidate >= start_day]
        for candidate in future_candidates:
            if len(selected) >= needed:
                break
            if candidate not in selected:
                selected.append(candidate)
        return set(selected[:needed])

    @staticmethod
    def _select_month_rest_plan_days(candidates: list[int], required_days: int) -> list[int]:
        if required_days <= 0 or not candidates:
            return []
        if len(candidates) <= required_days:
            return list(candidates)

        selected: list[int] = []
        previous_index = -1
        for slot in range(required_days):
            raw_index = math.floor((slot + 1) * (len(candidates) + 1) / (required_days + 1)) - 1
            remaining_slots = required_days - slot - 1
            max_index = len(candidates) - remaining_slots - 1
            index = min(max_index, max(previous_index + 1, raw_index))
            selected.append(candidates[index])
            previous_index = index
        return selected

    def _month_rest_blocked_days(self, rules: list[dict[str, Any]]) -> set[int]:
        blocked: set[int] = set()
        for event_rule in self._rules(rules, "scheduled_event"):
            params = self._params(event_rule)
            if params.get("mode") == "stops":
                active_start = int(params["active_start_minute"])
                active_end = int(params["active_end_minute"])
                block_start = active_start
                stops = params.get("stops") if isinstance(params.get("stops"), list) else []
                for index, stop in enumerate(stops):
                    if isinstance(stop, dict):
                        block_start = min(block_start, self._scheduled_prepare_start(index, params, rules))
                for day in range(max(0, block_start // 1440), min(_MONTHLY_REST_SCORING_DAYS, active_end // 1440 + 1)):
                    blocked.add(day)
            else:
                start_minute = int(params["event_start_minute"])
                end_minute = int(params["stay_until_minute"])
                block_start = max(0, start_minute - 24 * 60)
                for day in range(max(0, block_start // 1440), min(_MONTHLY_REST_SCORING_DAYS, end_minute // 1440 + 1)):
                    blocked.add(day)
        return blocked

    def _month_day_rest_protected_today(
        self,
        records: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        current_minute: int,
    ) -> bool:
        day = current_minute // 1440
        for rule in self._month_day_rules(rules):
            if day in self._month_fallback_rest_days(rule, records, current_minute, rules):
                return True
        return False

    def _month_day_item_conflicts(self, start_minute: int, finish_minute: int, protected: list[tuple[int, str]]) -> bool:
        for day, mode in protected:
            day_start = day * 1440
            day_end = day_start + 1440
            if mode == "no_active":
                if self._interval_overlap(start_minute, finish_minute, day_start, day_end):
                    return True
            elif finish_minute // 1440 == day:
                return True
        return False

    def _month_day_rule_satisfied(self, records: list[dict[str, Any]], day: int, mode: str) -> bool:
        contexts = self._step_contexts(records)
        if mode == "no_active":
            day_start = day * 1440
            day_end = day_start + 1440
            for ctx in contexts:
                if ctx["action_name"] in {"take_order", "reposition"} and self._interval_overlap(
                    int(ctx["action_start"]), int(ctx["action_end"]), day_start, day_end
                ):
                    return False
            return True
        for ctx in contexts:
            if ctx["action_name"] == "take_order" and ctx["accepted"] and int(ctx["step_end"]) // 1440 == day:
                return False
        return True

    def _order_cadence_wait_minutes(self, rule: dict[str, Any], records: list[dict[str, Any]], current_minute: int) -> int:
        day = current_minute // 1440
        minute_of_day = current_minute % 1440
        order_count = self._accepted_order_count_by_action_day(records, day)
        max_orders = rule.get("max_orders_per_day")
        if max_orders is not None and order_count >= int(max_orders):
            return self._minutes_until_next_day(current_minute)
        first_order_before = rule.get("first_order_before_minute")
        if first_order_before is not None and order_count == 0 and minute_of_day >= int(first_order_before):
            return self._minutes_until_next_day(current_minute)
        return 0

    def _accepted_order_count_by_action_day(self, records: list[dict[str, Any]], day: int) -> int:
        count = 0
        for ctx in self._step_contexts(records):
            if ctx["action_name"] == "take_order" and ctx["accepted"] and int(ctx["action_start"]) // 1440 == day:
                count += 1
        return count

    def _history_records(self, driver_id: str) -> list[dict[str, Any]]:
        history = self._api.query_decision_history(driver_id, -1)
        records = history.get("records", []) if isinstance(history, dict) else []
        return records if isinstance(records, list) else []

    def _step_contexts(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        prev_end = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            try:
                query_cost = int(record.get("query_scan_cost_minutes", 0) or 0)
                action_cost = int(record.get("action_exec_cost_minutes", 0) or 0)
                step_end = int(result.get("simulation_progress_minutes", prev_end))
            except (TypeError, ValueError):
                continue
            action = record.get("action") if isinstance(record.get("action"), dict) else {}
            before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
            after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            action_start = prev_end + query_cost
            contexts.append(
                {
                    "action_name": str(action.get("action", "")).strip().lower(),
                    "accepted": bool(result.get("accepted", False)),
                    "step_start": prev_end,
                    "action_start": action_start,
                    "action_end": action_start + action_cost,
                    "step_end": step_end,
                    "before_lat": self._safe_float(before.get("lat")) or 0.0,
                    "before_lng": self._safe_float(before.get("lng")) or 0.0,
                    "after_lat": self._safe_float(after.get("lat")) or 0.0,
                    "after_lng": self._safe_float(after.get("lng")) or 0.0,
                }
            )
            prev_end = step_end
        return contexts

    @staticmethod
    def _minutes_until_next_day(current_minute: int) -> int:
        return max(1, (current_minute // 1440 + 1) * 1440 - current_minute)

    @staticmethod
    def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) < min(a_end, b_end)

    @staticmethod
    def _quiet_window_interval(rule: dict[str, Any], day: int) -> tuple[int, int]:
        start = day * 1440 + int(rule["start_minute"])
        end = day * 1440 + int(rule["end_minute"])
        if rule.get("crosses_midnight"):
            end += 1440
        return start, end

    def _estimate_quiet_window_penalty(self, start_minute: int, end_minute: int, rules: list[dict[str, Any]]) -> float:
        if end_minute <= start_minute:
            return 0.0
        total = 0.0
        start_day = max(0, start_minute // 1440 - 1)
        end_day = max(0, end_minute // 1440)
        for rule in rules:
            overlaps = 0
            for day in range(start_day, end_day + 1):
                window_start, window_end = self._quiet_window_interval(rule, day)
                if max(start_minute, window_start) < min(end_minute, window_end):
                    overlaps += 1
            penalty = overlaps * float(rule.get("penalty_amount", 0.0) or 0.0)
            penalty_cap = rule.get("penalty_cap")
            if penalty_cap is not None:
                penalty = min(penalty, float(penalty_cap))
            total += penalty
        return total

    def _minutes_until_next_quiet_window_start(self, rules: list[dict[str, Any]], current_minute: int) -> int:
        candidates: list[int] = []
        current_day = current_minute // 1440
        for rule in rules:
            for day in (current_day, current_day + 1):
                start, _ = self._quiet_window_interval(rule, day)
                if start > current_minute:
                    candidates.append(start - current_minute)
        return min(candidates) if candidates else _NO_FEASIBLE_CARGO_WAIT_MINUTES

    def _estimate_rest_conflict_penalty(self, item: dict[str, Any], rule: dict[str, Any], decision_time_min: int) -> float:
        finish_minutes = self._safe_float(item.get("finish_minutes"))
        if finish_minutes is None:
            return 0.0
        params = self._params(rule)
        rest_minutes = int(params["minutes"])
        penalty_amount = float(rule.get("penalty_amount", 0.0) or 0.0)
        if rest_minutes <= 0 or penalty_amount <= 0:
            return 0.0

        start_day = decision_time_min // 1440
        end_day = max(start_day, int(finish_minutes) // 1440)
        violation_days = 0
        for day in range(start_day + 1, end_day + 1):
            if not self._daily_rest_rule_applies(params, day):
                continue
            day_end = day * 1440 + 1440
            can_rest_after_order = int(finish_minutes) < day_end and day_end - int(finish_minutes) >= rest_minutes
            if not can_rest_after_order:
                violation_days += 1
        penalty = violation_days * penalty_amount
        penalty_cap = rule.get("penalty_cap")
        if penalty_cap is not None:
            penalty = min(penalty, float(penalty_cap))
        return penalty

    @staticmethod
    def _daily_rest_rule_applies(params: dict[str, Any], day: int) -> bool:
        return params.get("applies_on") != "weekday" or (_SIMULATION_EPOCH.weekday() + day) % 7 < 5

    def _estimate_order_net_income(self, item: dict[str, Any]) -> float:
        cargo = item.get("cargo", {}) if isinstance(item.get("cargo"), dict) else {}
        price = self._safe_float(cargo.get("price")) or 0.0
        pickup_km = self._safe_float(item.get("distance_km")) or 0.0
        haul_km = self._cargo_haul_distance_km(cargo)
        return price - (pickup_km + haul_km) * _COST_PER_KM

    def _cargo_haul_distance_km(self, cargo: dict[str, Any]) -> float:
        distance = self._safe_float(cargo.get("distance_km"))
        if distance is not None:
            return distance
        start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        lat1 = self._safe_float(start.get("lat"))
        lng1 = self._safe_float(start.get("lng"))
        lat2 = self._safe_float(end.get("lat"))
        lng2 = self._safe_float(end.get("lng"))
        if None in (lat1, lng1, lat2, lng2):
            return 0.0
        return self._haversine_km(float(lat1), float(lng1), float(lat2), float(lng2))

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        radius = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
        return 2.0 * radius * math.asin(math.sqrt(a))

    def _near_point(self, lat: float, lng: float, target_lat: float, target_lng: float, radius_km: float) -> bool:
        return self._haversine_km(lat, lng, target_lat, target_lng) <= radius_km

    def _select_fallback_cargo_id(self, reachable_items: list[dict[str, Any]]) -> str | None:
        if not reachable_items:
            return None

        def score(item: dict[str, Any]) -> tuple[float, float, float]:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            price = self._safe_float(cargo.get("price")) or 0.0
            wait_minutes = self._safe_float(item.get("wait_before_loading_minutes")) or 0.0
            distance_km = self._safe_float(item.get("distance_km")) or 0.0
            return (-price, wait_minutes, distance_km)

        best = min(reachable_items, key=score)
        cargo = best.get("cargo") if isinstance(best.get("cargo"), dict) else {}
        cargo_id = str(cargo.get("cargo_id", "")).strip()
        return cargo_id or None

    @staticmethod
    def _find_reachable_item(cargo_id: str, reachable_items: list[dict[str, Any]]) -> dict[str, Any] | None:
        for item in reachable_items:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            if str(cargo.get("cargo_id", "")).strip() == cargo_id:
                return item
        return None

    def _remember_city_goal_action(
        self,
        driver_id: str,
        rules: list[dict[str, Any]],
        action: dict[str, Any],
        reachable_items: list[dict[str, Any]],
        decision_time_min: int,
    ) -> None:
        if action.get("action") != "take_order":
            return
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        if not cargo_id:
            return
        item = next(
            (
                candidate
                for candidate in reachable_items
                if str(((candidate.get("cargo") or {}).get("cargo_id", ""))).strip() == cargo_id
            ),
            None,
        )
        if item is None:
            return
        day = decision_time_min // 1440
        driver_days = self._city_goal_days_by_driver.setdefault(driver_id, {})
        for rule in self._rules(rules, "cargo_city_day_goal"):
            params = self._params(rule)
            if self._cargo_city_matches(item, params):
                driver_days.setdefault(self._city_goal_key(params), set()).add(day)

    def _is_low_market_for_driver(self, driver_id: str, sample: dict[str, Any]) -> bool:
        history = self._market_history_by_driver.get(driver_id, [])
        scores = [float(item.get("score", 0.0) or 0.0) for item in history]
        if len(scores) < _MARKET_HISTORY_MIN_SAMPLES:
            return False
        threshold = self._percentile(scores, _MARKET_LOW_QUANTILE)
        return float(sample.get("score", 0.0) or 0.0) <= threshold

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(math.floor((len(ordered) - 1) * quantile))))
        return ordered[idx]

    def _distance_rules(self, rules: list[dict[str, Any]]) -> dict[str, float]:
        distance_rules: dict[str, float] = {}
        for rule in self._rules(rules, "distance_limit"):
            params = self._params(rule)
            if params.get("hard") is False:
                continue
            metric = params.get("metric")
            key = {
                "haul": "max_haul_km",
                "pickup_deadhead": "max_pickup_km",
                "month_deadhead": "max_month_deadhead_km",
            }.get(str(metric))
            if key is None:
                continue
            self._set_min_rule(distance_rules, key, float(params["max_km"]))
        return distance_rules

    @staticmethod
    def _set_min_rule(rules: dict[str, float], key: str, value: float) -> None:
        current = rules.get(key)
        rules[key] = value if current is None else min(current, value)

    @staticmethod
    def _cargo_name(item: dict[str, Any]) -> str:
        cargo = item.get("cargo", {})
        return str(cargo.get("cargo_name", "") if isinstance(cargo, dict) else "").strip()

    def _cargo_city_matches(self, item: dict[str, Any], params: dict[str, Any]) -> bool:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        start_city = str(start.get("city", "") or "")
        end_city = str(end.get("city", "") or "")
        keywords = [str(item).strip() for item in params.get("city_keywords", []) if str(item).strip()]
        applies_to = str(params.get("applies_to", "pickup_or_dropoff") or "pickup_or_dropoff")
        if applies_to == "pickup":
            return self._city_text_matches(start_city, keywords)
        if applies_to == "dropoff":
            return self._city_text_matches(end_city, keywords)
        return self._city_text_matches(start_city, keywords) or self._city_text_matches(end_city, keywords)

    def _cargo_city_filter_matches(self, item: dict[str, Any], params: dict[str, Any], decision_time_min: int) -> bool:
        if not self._cargo_city_matches(item, params):
            return False
        active_ranges = params.get("active_ranges")
        if not isinstance(active_ranges, list) or not active_ranges:
            return True
        finish_minute = self._safe_int(item.get("finish_minutes"))
        if finish_minute is None:
            finish_minute = decision_time_min
        end_minute = max(decision_time_min + 1, int(finish_minute))
        return any(
            self._interval_overlap(
                decision_time_min,
                end_minute,
                int(active_range.get("start_minute", 0)),
                int(active_range.get("end_minute", 0)),
            )
            for active_range in active_ranges
            if isinstance(active_range, dict)
        )

    @staticmethod
    def _city_text_matches(city: str, keywords: list[str]) -> bool:
        return any(keyword in city for keyword in keywords)

    @staticmethod
    def _city_goal_key(params: dict[str, Any]) -> str:
        keywords = ",".join(sorted(str(item).strip() for item in params.get("city_keywords", []) if str(item).strip()))
        return f"{params.get('applies_to', 'pickup_or_dropoff')}:{keywords}:{params.get('required_days')}"

    def _month_day_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        month_rules: list[dict[str, Any]] = []
        for rule in self._rules(rules, "month_day_rest"):
            params = self._params(rule)
            month_rules.append({"required_days": int(params["required_days"]), "mode": str(params["mode"])})
        return month_rules

    def _order_cadence_rule(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        cadence: dict[str, Any] = {}
        for rule in self._rules(rules, "order_cadence"):
            params = self._params(rule)
            first = params.get("first_order_before_minute")
            if first is not None:
                cadence["first_order_before_minute"] = min(int(first), int(cadence.get("first_order_before_minute", first)))
            max_orders = params.get("max_orders_per_day")
            if max_orders is not None:
                cadence["max_orders_per_day"] = min(int(max_orders), int(cadence.get("max_orders_per_day", max_orders)))
        return cadence

    def _quiet_window_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        quiet_rules: list[dict[str, Any]] = []
        for rule in self._rules(rules, "quiet_window"):
            params = self._params(rule)
            quiet_rules.append(
                {
                    "start_minute": int(params["start_minute"]),
                    "end_minute": int(params["end_minute"]),
                    "crosses_midnight": bool(params.get("crosses_midnight")),
                    "penalty_amount": rule.get("penalty_amount", 0.0),
                    "penalty_cap": rule.get("penalty_cap"),
                }
            )
        return quiet_rules

    def _first_rule_params(self, rules: list[dict[str, Any]], rule_type: str) -> dict[str, Any] | None:
        rule = self._first_rule(rules, rule_type)
        return self._params(rule) if rule is not None else None

    @staticmethod
    def _first_rule(rules: list[dict[str, Any]], rule_type: str) -> dict[str, Any] | None:
        for rule in rules:
            if rule.get("rule_type") == rule_type:
                return rule
        return None

    @staticmethod
    def _rules(rules: list[dict[str, Any]], rule_type: str) -> list[dict[str, Any]]:
        return [rule for rule in rules if rule.get("rule_type") == rule_type]

    @staticmethod
    def _params(rule: dict[str, Any]) -> dict[str, Any]:
        params = rule.get("params")
        return params if isinstance(params, dict) else {}

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pickup_minutes(distance_km: float) -> int:
        if distance_km <= 1e-6:
            return 0
        return max(1, int(math.ceil((distance_km / _REPOSITION_SPEED_KM_PER_HOUR) * 60.0)))
