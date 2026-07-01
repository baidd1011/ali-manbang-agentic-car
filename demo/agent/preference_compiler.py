"""LLM-backed compiler for visible driver preferences.

Compilation is intentionally split into two model calls:
1. classify the preference into a known business rule type;
2. extract raw fields for that type only.

The model never needs to calculate simulation minute offsets. It returns
human-readable dates, clocks, coordinates, and numbers; this module validates
and normalizes them deterministically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_LOW_CONFIDENCE_THRESHOLD = 0.55
_HIGH_RISK_PENALTY_AMOUNT = 500.0
_HIGH_RISK_PENALTY_CAP = 3000.0
_MEDIUM_RISK_PENALTY_AMOUNT = 100.0
_MEDIUM_RISK_PENALTY_CAP = 1000.0

ALLOWED_RULE_TYPES = {
    "daily_rest",
    "quiet_window",
    "reject_cargo_category",
    "cargo_city_filter",
    "cargo_city_day_goal",
    "distance_limit",
    "month_day_rest",
    "order_cadence",
    "location_bounds",
    "location_exclusion_circle",
    "must_take_cargo",
    "daily_home",
    "scheduled_event",
    "point_visit",
    "unknown",
}


class PreferenceCompiler:
    """Compile visible preference objects into validated rule dictionaries."""

    def __init__(self, api: SimulationApiPort, logger: logging.Logger | None = None) -> None:
        self._api = api
        self._logger = logger or logging.getLogger("agent.preference_compiler")
        self._cache: dict[str, dict[str, Any]] = {}

    def compile(self, driver_id: str, preferences: Any) -> list[dict[str, Any]]:
        if not isinstance(preferences, list):
            return []
        rules: list[dict[str, Any]] = []
        for preference in preferences:
            key = self._cache_key(preference)
            cached = self._cache.get(key)
            if cached is not None:
                self._log_audit(driver_id, cached, cache_hit=True)
                rules.append(dict(cached))
                continue
            rule = self._compile_with_retry(preference)
            self._cache[key] = rule
            self._log_audit(driver_id, rule, cache_hit=False)
            rules.append(dict(rule))
        return rules

    def _compile_with_retry(self, preference: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        risk_level = self._risk_level(preference)
        if risk_level == "high":
            try:
                self._logger.info("preference compile high_risk_thinking")
                return self._compile_once(preference, attempt=0, enable_thinking=True, risk_level=risk_level)
            except Exception as exc:  # pragma: no cover - logged for runtime diagnosis
                last_error = exc
                self._logger.warning("preference compile high_risk_thinking failed error=%s", exc)
        for attempt in range(2):
            try:
                return self._compile_once(preference, attempt=attempt, enable_thinking=False, risk_level=risk_level)
            except Exception as exc:  # pragma: no cover - logged for runtime diagnosis
                last_error = exc
                self._logger.warning("preference compile failed attempt=%s error=%s", attempt + 1, exc)
        try:
            self._logger.info("preference compile fallback_thinking")
            return self._compile_once(preference, attempt=2, enable_thinking=True, risk_level=risk_level)
        except Exception as exc:  # pragma: no cover - logged for runtime diagnosis
            last_error = exc
            self._logger.warning("preference compile thinking fallback failed error=%s", exc)
        return self._unknown_rule(preference, f"compile_failed: {last_error}", risk_level=risk_level)

    def _compile_once(self, preference: Any, attempt: int, *, enable_thinking: bool, risk_level: str) -> dict[str, Any]:
        rule_type, confidence = self._classify(preference, attempt, enable_thinking=enable_thinking)
        if rule_type == "unknown":
            raise ValueError("classified_unknown")
        params = self._extract_fields(preference, rule_type, attempt, enable_thinking=enable_thinking)
        status = "thinking_compiled" if enable_thinking else "compiled"
        return self._validate_rule(
            rule_type,
            params,
            preference,
            confidence=confidence,
            compile_status=status,
            compile_error=None,
            risk_level=risk_level,
        )

    def _classify(self, preference: Any, attempt: int, *, enable_thinking: bool) -> tuple[str, float | None]:
        data = self._call_model(
            system_prompt=self._classification_prompt(),
            user_payload={
                "preference": preference,
                "attempt": attempt + 1,
                "task": "Classify this one visible preference into exactly one rule_type.",
            },
            enable_thinking=enable_thinking,
        )
        rule_type = str(data.get("rule_type", "")).strip()
        if rule_type not in ALLOWED_RULE_TYPES:
            raise ValueError(f"unsupported rule_type: {rule_type}")
        confidence = self._optional_float(data.get("confidence"))
        if confidence is not None and confidence < _LOW_CONFIDENCE_THRESHOLD:
            return "unknown", confidence
        return rule_type, confidence

    def _extract_fields(self, preference: Any, rule_type: str, attempt: int, *, enable_thinking: bool) -> dict[str, Any]:
        data = self._call_model(
            system_prompt=self._field_prompt(rule_type),
            user_payload={
                "preference": preference,
                "rule_type": rule_type,
                "attempt": attempt + 1,
                "task": "Extract raw fields only for this rule_type. Do not calculate simulation minute offsets.",
            },
            enable_thinking=enable_thinking,
        )
        params = data.get("params") if isinstance(data.get("params"), dict) else data
        if not isinstance(params, dict):
            raise ValueError("extracted params must be object")
        return params

    def _call_model(self, *, system_prompt: str, user_payload: dict[str, Any], enable_thinking: bool) -> dict[str, Any]:
        response = self._api.model_chat_completion(
            {
                "enable_thinking": enable_thinking,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            }
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty content")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("model response must be object")
        return data

    def _validate_rule(
        self,
        rule_type: str,
        params: dict[str, Any],
        preference: Any,
        *,
        confidence: float | None,
        compile_status: str,
        compile_error: str | None,
        risk_level: str,
    ) -> dict[str, Any]:
        if rule_type not in ALLOWED_RULE_TYPES:
            raise ValueError(f"unsupported rule_type: {rule_type}")
        if rule_type == "unknown":
            return self._unknown_rule(
                preference,
                "classified_unknown",
                confidence=confidence,
                compile_status="unknown",
                risk_level=risk_level,
            )
        normalized = self._normalize_params(rule_type, params, preference)
        return self._attach_compile_metadata(
            self._attach_source({"rule_type": rule_type, "params": normalized}, preference),
            confidence=confidence,
            compile_status=compile_status,
            compile_error=compile_error,
            risk_level=risk_level,
        )

    def _normalize_params(self, rule_type: str, params: dict[str, Any], preference: Any) -> dict[str, Any]:
        if rule_type == "daily_rest":
            minutes = self._optional_int(params.get("minutes"))
            if minutes is None:
                minutes = self._positive_float(params.get("hours")) * 60.0
            return {
                "minutes": self._positive_int(minutes),
                "required_count": self._positive_int(params.get("required_count"), default=1),
                "applies_on": self._enum(params.get("applies_on"), {"daily", "weekday"}, "daily"),
            }

        if rule_type == "quiet_window":
            start_minute = self._clock_to_minute(params.get("start_clock", params.get("start_minute")))
            end_minute = self._clock_to_minute(params.get("end_clock", params.get("end_minute")))
            return {
                "start_minute": start_minute,
                "end_minute": end_minute,
                "crosses_midnight": self._bool(params.get("crosses_midnight"), end_minute <= start_minute),
            }

        if rule_type == "reject_cargo_category":
            categories = self._string_list(params.get("categories"))
            if not categories:
                raise ValueError("reject_cargo_category.categories required")
            return {"categories": categories, "hard": self._bool(params.get("hard"), True)}

        if rule_type == "cargo_city_filter":
            city_keywords = self._string_list(params.get("city_keywords"))
            if not city_keywords:
                raise ValueError("cargo_city_filter.city_keywords required")
            normalized = {
                "city_keywords": city_keywords,
                "applies_to": self._enum(params.get("applies_to"), {"pickup", "dropoff", "pickup_or_dropoff"}, "pickup_or_dropoff"),
                "hard": self._bool(params.get("hard"), True),
            }
            active_ranges = self._active_time_ranges(params)
            if active_ranges:
                normalized["active_ranges"] = active_ranges
            return normalized

        if rule_type == "cargo_city_day_goal":
            city_keywords = self._string_list(params.get("city_keywords"))
            if not city_keywords:
                raise ValueError("cargo_city_day_goal.city_keywords required")
            normalized = {
                "city_keywords": city_keywords,
                "applies_to": self._enum(params.get("applies_to"), {"pickup", "dropoff", "pickup_or_dropoff"}, "pickup_or_dropoff"),
                "required_days": self._positive_int(params.get("required_days")),
                "radius_km": self._positive_float(params.get("radius_km"), default=1.0),
            }
            point = self._optional_point(params, "target")
            if point is not None:
                normalized["target_lat"] = point[0]
                normalized["target_lng"] = point[1]
            return normalized

        if rule_type == "distance_limit":
            return {
                "metric": self._enum(params.get("metric"), {"haul", "pickup_deadhead", "month_deadhead"}, "haul"),
                "max_km": self._positive_float(params.get("max_km")),
                "hard": self._bool(params.get("hard"), True),
            }

        if rule_type == "month_day_rest":
            return {
                "required_days": self._positive_int(params.get("required_days")),
                "mode": self._enum(params.get("mode"), {"no_order", "no_active"}, "no_active"),
            }

        if rule_type == "order_cadence":
            normalized: dict[str, Any] = {}
            first_order_before = params.get("first_order_before_clock", params.get("first_order_before_minute"))
            max_orders = self._optional_int(params.get("max_orders_per_day"))
            if first_order_before is not None:
                normalized["first_order_before_minute"] = self._clock_to_minute(first_order_before)
            if max_orders is not None:
                normalized["max_orders_per_day"] = self._positive_int(max_orders)
            if not normalized:
                raise ValueError("order_cadence needs at least one field")
            return normalized

        if rule_type == "location_bounds":
            lat_min = self._lat(params.get("lat_min"))
            lat_max = self._lat(params.get("lat_max"))
            lng_min = self._lng(params.get("lng_min"))
            lng_max = self._lng(params.get("lng_max"))
            return {
                "lat_min": min(lat_min, lat_max),
                "lat_max": max(lat_min, lat_max),
                "lng_min": min(lng_min, lng_max),
                "lng_max": max(lng_min, lng_max),
            }

        if rule_type == "location_exclusion_circle":
            center = self._point(params, "center")
            return {
                "center_lat": center[0],
                "center_lng": center[1],
                "radius_km": self._positive_float(params.get("radius_km")),
            }

        if rule_type == "must_take_cargo":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("must_take_cargo.cargo_id required")
            pickup = self._point(params, "pickup")
            start_value = params.get("cargo_available_time", params.get("active_start_time"))
            end_value = params.get("preference_end_time", params.get("active_end_time"))
            active_start = self._wall_time_to_minute(start_value)
            active_end = self._wall_time_to_minute(end_value, fallback=self._preference_end_time(preference))
            if active_end <= active_start:
                raise ValueError("must_take_cargo active_end must be after active_start")
            return {
                "cargo_id": cargo_id,
                "pickup_lat": pickup[0],
                "pickup_lng": pickup[1],
                "active_start_minute": active_start,
                "active_end_minute": active_end,
            }

        if rule_type == "daily_home":
            home = self._point(params, "home")
            normalized = {
                "home_lat": home[0],
                "home_lng": home[1],
                "radius_km": self._positive_float(params.get("radius_km"), default=1.0),
                "deadline_minute": self._clock_to_minute(params.get("deadline_clock", params.get("deadline_minute"))),
            }
            quiet_start = params.get("quiet_start_clock", params.get("quiet_start_minute"))
            quiet_end = params.get("quiet_end_clock", params.get("quiet_end_minute"))
            if quiet_start is not None or quiet_end is not None:
                if quiet_start is None or quiet_end is None:
                    raise ValueError("daily_home quiet window needs both start and end")
                normalized["quiet_start_minute"] = self._clock_to_minute(quiet_start)
                normalized["quiet_end_minute"] = self._clock_to_minute(quiet_end)
            return normalized

        if rule_type == "scheduled_event":
            if isinstance(params.get("stops"), list):
                stops = self._normalize_scheduled_stops(params)
                if not stops:
                    raise ValueError("scheduled_event.stops required")
                active_start = self._wall_time_to_minute(
                    params.get("active_start_time", params.get("event_start_time")),
                    fallback=self._preference_start_time(preference),
                )
                active_end = self._wall_time_to_minute(
                    params.get("active_end_time", params.get("event_end_time")),
                    fallback=self._preference_end_time(preference),
                )
                if active_end <= active_start:
                    raise ValueError("scheduled_event active_end must be after active_start")
                return {
                    "mode": "stops",
                    "active_start_minute": active_start,
                    "active_end_minute": active_end,
                    "stops": stops,
                }
            pickup = self._point(params, "pickup")
            home = self._point(params, "home")
            return {
                "mode": "pickup_home",
                "event_start_minute": self._wall_time_to_minute(params.get("event_start_time", params.get("event_start_minute"))),
                "pickup_lat": pickup[0],
                "pickup_lng": pickup[1],
                "home_lat": home[0],
                "home_lng": home[1],
                "pickup_stay_minutes": self._positive_int(params.get("pickup_stay_minutes"), default=10),
                "home_deadline_minute": self._wall_time_to_minute(params.get("home_deadline_time", params.get("home_deadline_minute"))),
                "stay_until_minute": self._wall_time_to_minute(params.get("stay_until_time", params.get("stay_until_minute"))),
                "radius_km": self._positive_float(params.get("radius_km"), default=1.0),
            }

        if rule_type == "point_visit":
            target = self._point(params, "target")
            normalized = {
                "target_lat": target[0],
                "target_lng": target[1],
                "radius_km": self._positive_float(params.get("radius_km"), default=1.0),
                "required_days": self._positive_int(params.get("required_days")),
            }
            place_keywords = self._string_list(params.get("place_keywords"))
            if place_keywords:
                normalized["place_keywords"] = place_keywords
            return normalized

        raise ValueError(f"missing normalizer for {rule_type}")

    def _normalize_scheduled_stops(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        stops: list[dict[str, Any]] = []
        for raw in params.get("stops") or []:
            if not isinstance(raw, dict):
                continue
            stop: dict[str, Any] = {
                "name": str(raw.get("name", "") or "").strip(),
                "place_keywords": self._string_list(raw.get("place_keywords")),
                "radius_km": self._positive_float(raw.get("radius_km"), default=1.0),
            }
            point = self._optional_point(raw, "target")
            if point is None:
                point = self._optional_point(raw, "point")
            if point is not None:
                stop["lat"] = point[0]
                stop["lng"] = point[1]
            if not stop["place_keywords"] and stop["name"]:
                stop["place_keywords"] = [stop["name"]]
            if "arrive_after_time" in raw or "arrive_after_minute" in raw:
                stop["arrive_after_minute"] = self._wall_time_to_minute(raw.get("arrive_after_time", raw.get("arrive_after_minute")))
            if "deadline_time" in raw or "deadline_minute" in raw:
                stop["deadline_minute"] = self._wall_time_to_minute(raw.get("deadline_time", raw.get("deadline_minute")))
            if "stay_until_time" in raw or "stay_until_minute" in raw:
                stop["stay_until_minute"] = self._wall_time_to_minute(raw.get("stay_until_time", raw.get("stay_until_minute")))
            stay_minutes = self._optional_int(raw.get("stay_minutes"))
            if stay_minutes is not None:
                stop["stay_minutes"] = self._positive_int(stay_minutes)
            if "lat" not in stop and not stop["place_keywords"]:
                raise ValueError("scheduled stop needs point or place_keywords")
            stops.append(stop)
        return stops

    def _active_time_ranges(self, params: dict[str, Any]) -> list[dict[str, int]]:
        ranges: list[dict[str, int]] = []
        if "active_start_time" in params or "active_end_time" in params:
            start = self._wall_time_to_minute(params.get("active_start_time"))
            end = self._wall_time_to_minute(params.get("active_end_time"))
            if end <= start:
                raise ValueError("active_end_time must be after active_start_time")
            ranges.append({"start_minute": start, "end_minute": end})
        for value in self._string_list(params.get("active_dates")):
            start = self._wall_time_to_minute(value)
            ranges.append({"start_minute": start, "end_minute": start + 1440})
        return ranges

    def _attach_source(self, rule: dict[str, Any], preference: Any) -> dict[str, Any]:
        if isinstance(preference, dict):
            content = str(preference.get("content", "") or "")
            penalty_amount = self._optional_float(preference.get("penalty_amount"))
            penalty_cap = self._optional_float(preference.get("penalty_cap"))
            source = {
                "content": content,
                "start_time": preference.get("start_time"),
                "end_time": preference.get("end_time"),
            }
        else:
            content = str(preference or "")
            penalty_amount = None
            penalty_cap = None
            source = {"content": content}
        rule["source_content"] = content
        rule["source"] = source
        rule["penalty_amount"] = penalty_amount or 0.0
        rule["penalty_cap"] = penalty_cap
        return rule

    def _unknown_rule(
        self,
        preference: Any,
        reason: str,
        *,
        confidence: float | None = None,
        compile_status: str = "failed",
        risk_level: str | None = None,
    ) -> dict[str, Any]:
        return self._attach_compile_metadata(
            self._attach_source({"rule_type": "unknown", "params": {"reason": reason}}, preference),
            confidence=confidence,
            compile_status=compile_status,
            compile_error=reason,
            risk_level=risk_level or self._risk_level(preference),
        )

    @staticmethod
    def _attach_compile_metadata(
        rule: dict[str, Any],
        *,
        confidence: float | None,
        compile_status: str,
        compile_error: str | None,
        risk_level: str,
    ) -> dict[str, Any]:
        rule["compile_confidence"] = confidence
        rule["compile_status"] = compile_status
        rule["compile_error"] = compile_error
        rule["risk_level"] = risk_level
        return rule

    def _log_audit(self, driver_id: str, rule: dict[str, Any], *, cache_hit: bool) -> None:
        params = rule.get("params") if isinstance(rule.get("params"), dict) else {}
        self._logger.info(
            "preference compile audit cache_hit=%s driver_id=%s rule_type=%s confidence=%s risk_level=%s status=%s params=%s content=%s",
            cache_hit,
            driver_id,
            rule.get("rule_type"),
            rule.get("compile_confidence"),
            rule.get("risk_level"),
            rule.get("compile_status"),
            json.dumps(params, ensure_ascii=False, sort_keys=True),
            self._short_text(str(rule.get("source_content", "") or "")),
        )

    @classmethod
    def _risk_level(cls, preference: Any) -> str:
        penalty_amount = None
        penalty_cap = None
        if isinstance(preference, dict):
            penalty_amount = cls._optional_float(preference.get("penalty_amount"))
            penalty_cap = cls._optional_float(preference.get("penalty_cap"))
        amount = float(penalty_amount or 0.0)
        cap = float(penalty_cap or 0.0)
        if amount >= _HIGH_RISK_PENALTY_AMOUNT or cap >= _HIGH_RISK_PENALTY_CAP:
            return "high"
        if penalty_cap is None and amount >= _MEDIUM_RISK_PENALTY_AMOUNT:
            return "high"
        if amount >= _MEDIUM_RISK_PENALTY_AMOUNT or cap >= _MEDIUM_RISK_PENALTY_CAP:
            return "medium"
        return "low"

    @staticmethod
    def _short_text(text: str, limit: int = 160) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return f"{clean[:limit]}..."

    @staticmethod
    def _cache_key(preference: Any) -> str:
        if isinstance(preference, dict):
            data = {
                "content": preference.get("content"),
                "start_time": preference.get("start_time"),
                "end_time": preference.get("end_time"),
                "penalty_amount": preference.get("penalty_amount"),
                "penalty_cap": preference.get("penalty_cap"),
            }
        else:
            data = {"content": str(preference or "")}
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _classification_prompt() -> str:
        examples = [
            ["每天至少连续停车休息满8小时。", "daily_rest"],
            ["我这人熬不住连轴转，每天至少连续停车熄火休息满8小时。", "daily_rest"],
            ["每天至少有一段连着停车歇满4小时（真熄火歇脚）。", "daily_rest"],
            ["别让我一天到晚跑个不停，每天怎么也得连着歇够五个钟头。", "daily_rest"],
            ["平日每天连续停车休息满4小时。", "daily_rest"],
            ["每天23点至次日6点不接单、不空车赶路。", "quiet_window"],
            ["凌晨两点到五点别派我接活，也别让我空车挪地方。", "quiet_window"],
            ["中午十二点到一点我要吃饭，不能接单也不能空跑。", "quiet_window"],
            ["零点以后到早上六点这段我得睡觉，车得停着熄火，雷打不动。", "quiet_window"],
            ["每天0点到6点必须停车睡觉，不要让我动。", "quiet_window"],
            ["夜里十二点到清晨六点车必须停着。", "quiet_window"],
            ["每天23点前车辆须在自家位置（22.72，114.12）一公里内；当天23点至次日8点不接单、不空跑。", "daily_home"],
            ["不接货源品类为「蔬菜」的订单。", "reject_cargo_category"],
            ["尽量不拉货源品类为「冷链生鲜」的订单。", "reject_cargo_category"],
            ["化工塑料和煤炭矿产这两类活看到就推掉。", "reject_cargo_category"],
            ["家电家具能不接就不接，实在没别的再说。", "reject_cargo_category"],
            ["装货地或卸货地在清远的货，我一律不接。", "cargo_city_filter"],
            ["只要货在佛山装或者送进佛山，三月四号五号都别给我。", "cargo_city_filter"],
            ["中山方向我不碰，起点终点沾上中山都算。", "cargo_city_filter"],
            ["自然月里装货或卸货在南海的货，起码接够四个不同的日子。", "cargo_city_day_goal"],
            ["这个月至少四天要跑到江门相关的货，一天多单也只算一天。", "cargo_city_day_goal"],
            ["单笔货装货点至卸货点距离不得超过150公里。", "distance_limit"],
            ["接单后赴装货点空驶距离不得超过50公里。", "distance_limit"],
            ["一个月空驶赶路里程总和不得超过100公里。", "distance_limit"],
            ["去装货那段空车路别超过五十五公里。", "distance_limit"],
            ["装到卸这趟别超过一百八十公里。", "distance_limit"],
            ["自然月内至少要有2天完全歇着：不接单也不空车乱跑。", "month_day_rest"],
            ["这月给我留三整天陪家里人，全天别排活。", "month_day_rest"],
            ["同一天接单不得超过3单。", "order_cadence"],
            ["只要这天接了单，首单开工不得晚于当天中午12点。", "order_cadence"],
            ["一天最多跑三票，多了我吃不消。", "order_cadence"],
            ["跑车或停车时，车辆位置须始终在指定范围内（北纬22.10至22.80，东经113.20至114.10）。", "location_bounds"],
            ["我就在这个经纬度框里干活，车不许出去。", "location_bounds"],
            ["车辆不得进入以（22.90，113.80）为圆心、半径20公里的区域。", "location_exclusion_circle"],
            ["离 22.90,113.80 周围二十公里都别进去。", "location_exclusion_circle"],
            ["指定熟货源编号A77881：装货地（24.20，112.90）；上架时间：2026-03-08 14:43:36。", "must_take_cargo"],
            ["老客户那票 B13579 必须接，错过要赔信任。", "must_take_cargo"],
            ["2026年3月18日10:00，家中急事：须先到（22.91，113.66）接上家人，再返回老家（22.88，113.62）。", "scheduled_event"],
            ["三月十二号东城老仓库清库存，当天得到东城区停一趟，花两小时把数目对清楚。", "scheduled_event"],
            ["三月二十一号上午先过东城区仓库，中午十二点前赶到西江镇（23.42，112.36）赴宴到下午两点。", "scheduled_event"],
            ["十号家里出事，先接人再回家，事情没办完之前车得停在家。", "scheduled_event"],
            ["自然月内至少5个不同的自然日到过（23.13，113.26）一公里内。", "point_visit"],
            ["这个月至少五天去过 23.13,113.26 附近一公里。", "point_visit"],
            ["天气不好就别让我出车。", "unknown"],
            ["别跑太远也别太近，自己看着办。", "unknown"],
            ["路上如果感觉不顺就停一下。", "unknown"],
        ]
        return (
            "你是司机偏好分类器。只输出JSON对象：{\"rule_type\":枚举值,\"confidence\":0到1}。"
            "只允许 rule_type 为："
            "daily_rest, quiet_window, reject_cargo_category, cargo_city_filter, cargo_city_day_goal, distance_limit, month_day_rest, "
            "order_cadence, location_bounds, location_exclusion_circle, must_take_cargo, "
            "daily_home, scheduled_event, point_visit, unknown。"
            "不要抽字段，不要换算时间。"
            "业务语义在这些枚举内、且具备可执行字段时，即使表达很口语、很长、带情绪或先讲原因，也应选最接近的已知 rule_type。"
            "如果偏好依赖外部不可见事实、主观感受，或缺少执行所必需的地点/时间/阈值/对象，分类为 unknown。"
            "凡是表达每天/每日/平日需要连续停车、停车熄火、歇脚、休息满N小时，且没有指定固定起止时钟，分类为 daily_rest；"
            "不要因为口语化前缀如熬不住连轴转而分类为 unknown。"
            "只要指定固定起止时钟或固定时段，例如零点到六点、凌晨两点到五点、中午十二点到一点、23点到次日6点，"
            "并要求不接单、不空驶、停车、熄火、睡觉或不要动车，均分类为 quiet_window，不要分类为 daily_rest；"
            "只有明确要求每天某时前回到家/自家位置，才分类为 daily_home；"
            "装货地/卸货地命中某城市或区域就不接，分类为 cargo_city_filter，不要转成坐标圆形禁区；"
            "自然月内要求接够某城市/区域装货或卸货的不同日子，分类为 cargo_city_day_goal；"
            "指定日期要到某地办事、停留、赴宴、接人回家，分类为 scheduled_event。"
            f"few-shot examples={json.dumps(examples, ensure_ascii=False)}"
        )

    @staticmethod
    def _field_prompt(rule_type: str) -> str:
        common = (
            "只抽取原文明确给出的字段；不要编造坐标、城市、日期、时间、距离或次数。"
            "如果原文使用三月/这个月，按 2026-03 解析。"
            "软偏好例如尽量/能不/不太想，hard=false；一律/不得/不接/不许/必须，hard=true。"
        )
        prompts = {
            "daily_rest": (
                common +
                "抽取 daily_rest 字段。只输出JSON："
                "{\"params\":{\"hours\":数字或null,\"minutes\":数字或null,\"required_count\":1,\"applies_on\":\"daily或weekday\"}}。"
                "例：'平日每天连续停车休息满4小时' -> {\"params\":{\"hours\":4,\"required_count\":1,\"applies_on\":\"weekday\"}}。"
            ),
            "quiet_window": (
                common +
                "抽取 quiet_window 字段。只输出JSON："
                "{\"params\":{\"start_clock\":\"HH:MM\",\"end_clock\":\"HH:MM\",\"crosses_midnight\":布尔}}。"
                "例：'每天23点至次日6点不接单、不空车赶路' -> "
                "{\"params\":{\"start_clock\":\"23:00\",\"end_clock\":\"06:00\",\"crosses_midnight\":true}}。"
                "例：'每天凌晨2点至5点不接单、不空驶' -> "
                "{\"params\":{\"start_clock\":\"02:00\",\"end_clock\":\"05:00\",\"crosses_midnight\":false}}。"
                "例：'零点以后到早上六点这段我得睡觉，车得停着熄火' -> "
                "{\"params\":{\"start_clock\":\"00:00\",\"end_clock\":\"06:00\",\"crosses_midnight\":false}}。"
            ),
            "reject_cargo_category": (
                common +
                "抽取 reject_cargo_category 字段。只输出JSON："
                "{\"params\":{\"categories\":[货源品类字符串],\"hard\":布尔}}。"
                "例：'不接货源品类为「化工塑料」或「煤炭矿产」的订单' -> "
                "{\"params\":{\"categories\":[\"化工塑料\",\"煤炭矿产\"],\"hard\":true}}。"
                "例：'尽量不拉货源品类为「冷链生鲜」的订单' -> "
                "{\"params\":{\"categories\":[\"冷链生鲜\"],\"hard\":false}}。"
            ),
            "cargo_city_filter": (
                common +
                "抽取 cargo_city_filter 字段。只输出JSON："
                "{\"params\":{\"city_keywords\":[城市或区域关键词],\"applies_to\":\"pickup或dropoff或pickup_or_dropoff\","
                "\"hard\":布尔,\"active_dates\":[\"YYYY-MM-DD\"]}}。"
                "装货地/起点/从某地发货 => pickup；卸货地/终点/送到某地 => dropoff；装货或卸货/货源在某地 => pickup_or_dropoff。"
                "如果原文限定具体日期，例如三月四号五号，只输出 active_dates；没有日期限制则省略 active_dates。"
                "不要把城市/区域文字限制改写成 location_exclusion_circle。"
                "例：'装货地或卸货地在清远的货，我一律不接' -> "
                "{\"params\":{\"city_keywords\":[\"清远\"],\"applies_to\":\"pickup_or_dropoff\",\"hard\":true}}。"
                "例：'三月四号五号不往佛山跑，也别派进佛山的货' -> "
                "{\"params\":{\"city_keywords\":[\"佛山\"],\"applies_to\":\"pickup_or_dropoff\",\"hard\":true,"
                "\"active_dates\":[\"2026-03-04\",\"2026-03-05\"]}}。"
            ),
            "cargo_city_day_goal": (
                common +
                "抽取 cargo_city_day_goal 字段。只输出JSON："
                "{\"params\":{\"city_keywords\":[城市或区域关键词],\"applies_to\":\"pickup或dropoff或pickup_or_dropoff\","
                "\"required_days\":数字,\"target\":{\"lat\":数字,\"lng\":数字},\"radius_km\":数字}}。"
                "自然月里接够某城市/区域装货或卸货的不同自然日属于本类型；有坐标就输出 target，没有坐标不要编造。"
                "例：'自然月里装货或卸货在南海的货，起码接够四个不同的日子，仓库（22.95，113.15）' -> "
                "{\"params\":{\"city_keywords\":[\"南海\"],\"applies_to\":\"pickup_or_dropoff\","
                "\"required_days\":4,\"target\":{\"lat\":22.95,\"lng\":113.15},\"radius_km\":1}}。"
            ),
            "distance_limit": (
                common +
                "抽取 distance_limit 字段。只输出JSON："
                "{\"params\":{\"metric\":\"haul或pickup_deadhead或month_deadhead\",\"max_km\":数字,\"hard\":true或false}}。"
                "装货点至卸货点距离 => haul；赴装货点空驶距离 => pickup_deadhead；"
                "一个月空驶赶路里程总和 => month_deadhead。"
                "出现不得、不能、超过就扣、上限等硬约束时 hard=true；只说尽量、最好、不太想时 hard=false。"
            ),
            "month_day_rest": (
                common +
                "抽取 month_day_rest 字段。只输出JSON："
                "{\"params\":{\"required_days\":数字,\"mode\":\"no_order或no_active\"}}。"
                "只有明确说空驶不计、可以空车移动、只是不能接单，才输出 no_order。"
                "只要出现整天/全天休息、停驶、停车、检修、保养、休假、完全歇着、别排活、不接单也不空车/不外跑，输出 no_active。"
                "拿不准时输出 no_active，偏好罚分优先于收益。"
            ),
            "order_cadence": (
                common +
                "抽取 order_cadence 字段。只输出JSON，可包含一个或两个字段："
                "{\"params\":{\"first_order_before_clock\":\"HH:MM\",\"max_orders_per_day\":数字}}。"
                "例：'首单不得晚于当天中午12点' -> {\"params\":{\"first_order_before_clock\":\"12:00\"}}。"
                "例：'同一天接单不得超过3单' -> {\"params\":{\"max_orders_per_day\":3}}。"
            ),
            "location_bounds": (
                common +
                "抽取 location_bounds 字段。只输出JSON："
                "{\"params\":{\"lat_min\":数字,\"lat_max\":数字,\"lng_min\":数字,\"lng_max\":数字}}。"
            ),
            "location_exclusion_circle": (
                common +
                "抽取 location_exclusion_circle 字段。只输出JSON："
                "{\"params\":{\"center\":{\"lat\":数字,\"lng\":数字},\"radius_km\":数字}}。"
            ),
            "must_take_cargo": (
                common +
                "抽取 must_take_cargo 字段。只输出JSON："
                "{\"params\":{\"cargo_id\":\"字符串\",\"pickup\":{\"lat\":数字,\"lng\":数字},"
                "\"cargo_available_time\":\"YYYY-MM-DD HH:MM:SS\",\"preference_end_time\":\"YYYY-MM-DD HH:MM:SS\"}}。"
                "不要输出 simulation minute。cargo_available_time 使用货源上架时间/可接时间；"
                "preference_end_time 使用偏好对象 end_time。"
                "例：'编号A77881，装货地（24.20，112.90）；上架时间：2026-03-08 14:43:36' 且 end_time 为 "
                "'2026-03-08 16:08:24' -> {\"params\":{\"cargo_id\":\"A77881\",\"pickup\":{\"lat\":24.20,\"lng\":112.90},"
                "\"cargo_available_time\":\"2026-03-08 14:43:36\",\"preference_end_time\":\"2026-03-08 16:08:24\"}}。"
            ),
            "daily_home": (
                common +
                "抽取 daily_home 字段。只输出JSON："
                "{\"params\":{\"home\":{\"lat\":数字,\"lng\":数字},\"radius_km\":数字,"
                "\"deadline_clock\":\"HH:MM\",\"quiet_start_clock\":\"HH:MM或省略\",\"quiet_end_clock\":\"HH:MM或省略\"}}。"
                "例：'每天23点前车辆须在自家位置（23.12，113.28）一公里内；当天23点至次日8点不接单、不空跑' -> "
                "{\"params\":{\"home\":{\"lat\":23.12,\"lng\":113.28},\"radius_km\":1,"
                "\"deadline_clock\":\"23:00\",\"quiet_start_clock\":\"23:00\",\"quiet_end_clock\":\"08:00\"}}。"
                "如果只说几点前回家，没有说明到几点不能离开，不要编造 quiet_start_clock/quiet_end_clock。"
                "例：'每天晚上十点之前到家，家在（23.40，113.16），两公里内算到' -> "
                "{\"params\":{\"home\":{\"lat\":23.40,\"lng\":113.16},\"radius_km\":2,\"deadline_clock\":\"22:00\"}}。"
                "如果没有自家/家/老家坐标，不要编造坐标。"
            ),
            "scheduled_event": (
                common +
                "抽取 scheduled_event 字段。优先输出通用 stops 结构，只输出JSON："
                "{\"params\":{\"active_start_time\":\"YYYY-MM-DD HH:MM:SS\",\"active_end_time\":\"YYYY-MM-DD HH:MM:SS\","
                "\"stops\":[{\"name\":\"地点名\",\"place_keywords\":[地点关键词],\"target\":{\"lat\":数字,\"lng\":数字},"
                "\"deadline_time\":\"YYYY-MM-DD HH:MM:SS\",\"stay_minutes\":数字,\"stay_until_time\":\"YYYY-MM-DD HH:MM:SS\","
                "\"radius_km\":数字}]}}。"
                "如果地点没有坐标但能用名称指代，例如'东城区仓库'，不要编造坐标，输出 place_keywords。"
                "指定日期全天办事，active_start_time 用当天 00:00:00，active_end_time 用当天 23:59:59。"
                "例：'三月十二号东城老仓库清库存，当天得到东城区停一趟，花两小时' -> "
                "{\"params\":{\"active_start_time\":\"2026-03-12 00:00:00\",\"active_end_time\":\"2026-03-12 23:59:59\","
                "\"stops\":[{\"name\":\"东城老仓库\",\"place_keywords\":[\"东城\",\"东城区\",\"仓库\"],"
                "\"stay_minutes\":120,\"radius_km\":1}]}}。"
                "例：'三月二十一号上午先过东城区仓库，中午十二点前赶到西江镇（23.42，112.36）赴宴到下午两点' -> "
                "{\"params\":{\"active_start_time\":\"2026-03-21 00:00:00\",\"active_end_time\":\"2026-03-21 23:59:59\","
                "\"stops\":[{\"name\":\"东城区仓库\",\"place_keywords\":[\"东城\",\"东城区\",\"仓库\"],\"radius_km\":1},"
                "{\"name\":\"西江镇\",\"target\":{\"lat\":23.42,\"lng\":112.36},"
                "\"deadline_time\":\"2026-03-21 12:00:00\",\"stay_until_time\":\"2026-03-21 14:00:00\",\"radius_km\":1}]}}。"
                "兼容旧接人回家结构也可以输出："
                "{\"params\":{\"event_start_time\":\"YYYY-MM-DD HH:MM:SS\",\"pickup\":{\"lat\":数字,\"lng\":数字},"
                "\"home\":{\"lat\":数字,\"lng\":数字},\"pickup_stay_minutes\":数字,"
                "\"home_deadline_time\":\"YYYY-MM-DD HH:MM:SS\",\"stay_until_time\":\"YYYY-MM-DD HH:MM:SS\",\"radius_km\":数字}}。"
                "例：'2026年3月18日10:00，须先到（22.91，113.66）接上家人，停留不少于10分钟，"
                "再返回老家（22.88，113.62）；须在2026年3月18日22:00前进家门，至少待到2026年3月20日22:00' -> "
                "{\"params\":{\"event_start_time\":\"2026-03-18 10:00:00\",\"pickup\":{\"lat\":22.91,\"lng\":113.66},"
                "\"home\":{\"lat\":22.88,\"lng\":113.62},\"pickup_stay_minutes\":10,"
                "\"home_deadline_time\":\"2026-03-18 22:00:00\",\"stay_until_time\":\"2026-03-20 22:00:00\",\"radius_km\":1}}。"
            ),
            "point_visit": (
                common +
                "抽取 point_visit 字段。只输出JSON："
                "{\"params\":{\"target\":{\"lat\":数字,\"lng\":数字},\"radius_km\":数字,\"required_days\":数字,"
                "\"place_keywords\":[地点关键词或省略]}}。"
                "例：'自然月内至少5个不同的自然日到过（23.13，113.26）一公里内' -> "
                "{\"params\":{\"target\":{\"lat\":23.13,\"lng\":113.26},\"radius_km\":1,\"required_days\":5}}。"
            ),
        }
        return prompts.get(rule_type, "输出 {\"params\":{}}。")

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _enum(value: Any, allowed: set[str], default: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else default

    @staticmethod
    def _bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return default

    @staticmethod
    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid float: {value}") from exc

    @classmethod
    def _positive_float(cls, value: Any, default: float | None = None) -> float:
        if value is None and default is not None:
            value = default
        number = cls._float(value)
        if number <= 0:
            raise ValueError(f"expected positive float: {value}")
        return number

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _positive_int(cls, value: Any, default: int | None = None) -> int:
        if value is None and default is not None:
            value = default
        number = cls._optional_int(value)
        if number is None or number <= 0:
            raise ValueError(f"expected positive int: {value}")
        return number

    @classmethod
    def _lat(cls, value: Any) -> float:
        number = cls._float(value)
        if number < -90.0 or number > 90.0 or abs(number) < 1e-9:
            raise ValueError(f"invalid latitude: {value}")
        return number

    @classmethod
    def _lng(cls, value: Any) -> float:
        number = cls._float(value)
        if number < -180.0 or number > 180.0 or abs(number) < 1e-9:
            raise ValueError(f"invalid longitude: {value}")
        return number

    @classmethod
    def _point(cls, params: dict[str, Any], prefix: str) -> tuple[float, float]:
        value = params.get(prefix)
        if isinstance(value, dict):
            lat_value = value.get("lat")
            lng_value = value.get("lng")
        else:
            lat_value = params.get(f"{prefix}_lat")
            lng_value = params.get(f"{prefix}_lng")
        return cls._lat(lat_value), cls._lng(lng_value)

    @classmethod
    def _optional_point(cls, params: dict[str, Any], prefix: str) -> tuple[float, float] | None:
        value = params.get(prefix)
        if isinstance(value, dict):
            lat_value = value.get("lat")
            lng_value = value.get("lng")
        else:
            lat_value = params.get(f"{prefix}_lat")
            lng_value = params.get(f"{prefix}_lng")
        if lat_value is None or lng_value is None:
            return None
        return cls._lat(lat_value), cls._lng(lng_value)

    @classmethod
    def _clock_to_minute(cls, value: Any) -> int:
        if isinstance(value, (int, float)):
            number = int(value)
            if 0 <= number <= 1439:
                return number
            raise ValueError(f"minute of day out of range: {value}")
        text = str(value or "").strip()
        if not text:
            raise ValueError("empty clock")
        cleaned = text.replace("点", ":").replace("时", ":").replace("：", ":")
        parts = [part for part in cleaned.split(":") if part != ""]
        if len(parts) == 1 and parts[0].isdigit():
            hour = int(parts[0])
            minute = 0
        elif len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            hour = int(parts[0])
            minute = int(parts[1])
        else:
            raise ValueError(f"invalid clock: {value}")
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"clock out of range: {value}")
        return hour * 60 + minute

    @classmethod
    def _wall_time_to_minute(cls, value: Any, fallback: Any = None) -> int:
        source = value if value not in (None, "") else fallback
        if isinstance(source, (int, float)):
            number = int(source)
            if number >= 0:
                return number
            raise ValueError(f"negative minute: {value}")
        text = str(source or "").strip()
        if not text:
            raise ValueError("empty wall time")
        normalized = (
            text.replace("年", "-")
            .replace("月", "-")
            .replace("日", " ")
            .replace("时", ":")
            .replace("点", ":")
            .replace("分", "")
            .replace("秒", "")
        )
        while "  " in normalized:
            normalized = normalized.replace("  ", " ")
        if normalized.count(":") == 1:
            normalized = f"{normalized}:00"
        try:
            dt = datetime.fromisoformat(normalized.replace(" ", "T"))
        except ValueError as exc:
            raise ValueError(f"invalid wall time: {value}") from exc
        minute = int((dt - _SIMULATION_EPOCH).total_seconds() // 60)
        if minute < 0:
            raise ValueError(f"wall time before simulation epoch: {value}")
        return minute

    @staticmethod
    def _preference_end_time(preference: Any) -> Any:
        return preference.get("end_time") if isinstance(preference, dict) else None

    @staticmethod
    def _preference_start_time(preference: Any) -> Any:
        return preference.get("start_time") if isinstance(preference, dict) else None
