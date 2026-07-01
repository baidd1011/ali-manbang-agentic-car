"""Self-learning hotspot reposition planner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_GRID_SIZE_DEGREES = 0.1
_COST_PER_KM = 1.5
_MARKET_HISTORY_MIN_SAMPLES = 6
_MARKET_HISTORY_MAX_SAMPLES = 200
_MARKET_LOW_QUANTILE = 0.25
_MIN_DRIVER_SAMPLES = 6
_MIN_BIN_SAMPLES = 3
_MAX_BINS_PER_DRIVER = 1500
_MAX_REPOSITION_KM = 220.0
_MIN_REPOSITION_KM = 8.0
_ARRIVAL_RADIUS_KM = 3.0
_DEBOUNCE_MINUTES = 360
_TOP_NETS_PER_BIN = 5
_PROFITABLE_COUNT_WEIGHT = 10.0
_MAX_PROFITABLE_WEIGHT_COUNT = 20
_STALE_PENALTY_PER_HOUR = 20.0
_MAX_STALE_PENALTY = 800.0
_NO_REACHABLE_MIN_TARGET_SCORE = 50.0
_LOW_MARKET_MIN_IMPROVEMENT = 200.0
_LOW_MARKET_MIN_RELATIVE_IMPROVEMENT = 0.2
_OBSERVED_ONLY_CONFIDENCE = 0.55
_MIN_OBSERVED_ONLY_OBSERVATIONS = 3
_MIN_OBSERVED_ONLY_BIN_SAMPLES = 5
_DESTINATION_VALUE_RADIUS_KM = 90.0
_DESTINATION_VALUE_MAX = 1000.0

HotspotKey = tuple[int, int, int]


@dataclass
class _HotspotStats:
    count: int = 0
    reachable_count: int = 0
    profitable_count: int = 0
    net_sum: float = 0.0
    pickup_km_sum: float = 0.0
    lat_sum: float = 0.0
    lng_sum: float = 0.0
    best_net: float = float("-inf")
    top_nets: list[float] = field(default_factory=list)
    last_seen_minute: int = 0

    def add(self, *, minute: int, lat: float, lng: float, net: float, pickup_km: float, reachable: bool) -> None:
        self.count += 1
        self.reachable_count += 1 if reachable else 0
        self.profitable_count += 1 if net > 0 else 0
        self.net_sum += net
        self.pickup_km_sum += pickup_km
        self.lat_sum += lat
        self.lng_sum += lng
        self.best_net = max(self.best_net, net)
        self.top_nets.append(net)
        self.top_nets.sort(reverse=True)
        del self.top_nets[_TOP_NETS_PER_BIN:]
        self.last_seen_minute = max(self.last_seen_minute, minute)

    @property
    def lat(self) -> float:
        return self.lat_sum / self.count if self.count else 0.0

    @property
    def lng(self) -> float:
        return self.lng_sum / self.count if self.count else 0.0

    @property
    def avg_pickup_km(self) -> float:
        return self.pickup_km_sum / self.count if self.count else 0.0

    @property
    def top_avg_net(self) -> float:
        return sum(self.top_nets) / len(self.top_nets) if self.top_nets else 0.0


class HotspotPlanner:
    """Learn per-driver cargo hotspots from already queried cargo samples."""

    def __init__(self) -> None:
        self._bins_by_driver: dict[str, dict[HotspotKey, _HotspotStats]] = {}
        self._total_samples_by_driver: dict[str, int] = {}
        self._observation_count_by_driver: dict[str, int] = {}
        self._market_history_by_driver: dict[str, list[float]] = {}
        self._last_reposition_by_driver: dict[str, dict[str, Any]] = {}
        self._suppressed_until_by_driver: dict[str, dict[HotspotKey, int]] = {}

    def observe(
        self,
        driver_id: str,
        current_minute: int,
        raw_items: Any,
        reachable_items: list[dict[str, Any]],
    ) -> None:
        if not isinstance(raw_items, list):
            return
        bins = self._bins_by_driver.setdefault(driver_id, {})
        observed = 0
        reachable_by_id = {
            self._cargo_id(item): item
            for item in reachable_items
            if isinstance(item, dict) and self._cargo_id(item)
        }
        seen_ids: set[str] = set()
        observed_items = list(raw_items)
        for item in reachable_items:
            if isinstance(item, dict) and self._cargo_id(item) not in seen_ids:
                observed_items.append(item)
        for item in observed_items:
            if not isinstance(item, dict):
                continue
            cargo_id = self._cargo_id(item)
            if cargo_id and cargo_id in seen_ids:
                continue
            if cargo_id:
                seen_ids.add(cargo_id)
            source_item = reachable_by_id.get(cargo_id, item)
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
            lat = self._safe_float(start.get("lat"))
            lng = self._safe_float(start.get("lng"))
            if lat is None or lng is None:
                continue
            pickup_km = self._safe_float(source_item.get("distance_km")) or 0.0
            net = self._estimate_local_order_net(cargo)
            hour = self._cargo_observation_hour(cargo, current_minute)
            key = self._hotspot_key(hour, float(lat), float(lng))
            stats = bins.setdefault(key, _HotspotStats())
            stats.add(
                minute=current_minute,
                lat=float(lat),
                lng=float(lng),
                net=net,
                pickup_km=pickup_km,
                reachable=cargo_id in reachable_by_id,
            )
            observed += 1
        if observed:
            self._total_samples_by_driver[driver_id] = self._total_samples_by_driver.get(driver_id, 0) + observed
            self._observation_count_by_driver[driver_id] = self._observation_count_by_driver.get(driver_id, 0) + 1
            self._trim_old_bins(driver_id)

    def remember_market_sample(self, driver_id: str, market_sample: dict[str, Any]) -> None:
        score = self._safe_float(market_sample.get("score"))
        if score is None:
            return
        history = self._market_history_by_driver.setdefault(driver_id, [])
        history.append(float(score))
        if len(history) > _MARKET_HISTORY_MAX_SAMPLES:
            del history[: len(history) - _MARKET_HISTORY_MAX_SAMPLES]

    def plan_reposition(
        self,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        current_market_sample: dict[str, Any],
        has_reachable_items: bool,
    ) -> dict[str, Any] | None:
        self._update_recent_reposition_feedback(
            driver_id=driver_id,
            current_minute=current_minute,
            current_lat=current_lat,
            current_lng=current_lng,
            has_reachable_items=has_reachable_items,
        )
        if self._total_samples_by_driver.get(driver_id, 0) < _MIN_DRIVER_SAMPLES:
            return None

        action_reason = "no_reachable"
        if has_reachable_items:
            if not self._is_low_market_for_driver(driver_id, current_market_sample):
                return None
            action_reason = "low_market"

        current_score = self._safe_float(current_market_sample.get("score")) or 0.0
        for candidate in self._ranked_candidates(
            driver_id,
            current_minute,
            current_lat,
            current_lng,
            allow_observed_only=not has_reachable_items and self._can_use_observed_only(driver_id),
        ):
            target_score = float(candidate["target_score"])
            if action_reason == "no_reachable":
                if target_score < _NO_REACHABLE_MIN_TARGET_SCORE:
                    continue
            else:
                required_gain = max(
                    _LOW_MARKET_MIN_IMPROVEMENT,
                    abs(current_score) * _LOW_MARKET_MIN_RELATIVE_IMPROVEMENT,
                )
                if target_score - current_score < required_gain:
                    continue
            return {
                "action": {
                    "action": "reposition",
                    "params": {"latitude": candidate["latitude"], "longitude": candidate["longitude"]},
                },
                "meta": {
                    **candidate,
                    "action_reason": action_reason,
                    "current_score": current_score,
                },
            }
        return None

    def remember_reposition(self, driver_id: str, plan: dict[str, Any], current_minute: int) -> None:
        action = plan.get("action") if isinstance(plan, dict) else None
        meta = plan.get("meta") if isinstance(plan, dict) else None
        if not isinstance(action, dict) or action.get("action") != "reposition" or not isinstance(meta, dict):
            return
        key = meta.get("hotspot_key")
        if not isinstance(key, tuple):
            return
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        lat = self._safe_float(params.get("latitude"))
        lng = self._safe_float(params.get("longitude"))
        if lat is None or lng is None:
            return
        self._last_reposition_by_driver[driver_id] = {
            "hotspot_key": key,
            "minute": current_minute,
            "latitude": float(lat),
            "longitude": float(lng),
        }

    def _ranked_candidates(
        self,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        allow_observed_only: bool,
    ) -> list[dict[str, Any]]:
        current_hour = self._hour_of_day(current_minute)
        current_hour_candidates = self._candidates_for_hours(
            driver_id=driver_id,
            hours=[current_hour],
            current_minute=current_minute,
            current_lat=current_lat,
            current_lng=current_lng,
            allow_observed_only=allow_observed_only,
        )
        candidates = current_hour_candidates
        if not candidates:
            candidates = self._candidates_for_hours(
                driver_id=driver_id,
                hours=[(current_hour - 1) % 24, current_hour, (current_hour + 1) % 24, (current_hour + 2) % 24],
                current_minute=current_minute,
                current_lat=current_lat,
                current_lng=current_lng,
                allow_observed_only=allow_observed_only,
            )
        candidates.sort(key=lambda item: float(item["target_score"]), reverse=True)
        return candidates

    def _candidates_for_hours(
        self,
        driver_id: str,
        hours: list[int],
        current_minute: int,
        current_lat: float,
        current_lng: float,
        allow_observed_only: bool,
    ) -> list[dict[str, Any]]:
        bins = self._bins_by_driver.get(driver_id, {})
        suppressed = self._suppressed_until_by_driver.get(driver_id, {})
        hour_set = set(hours)
        candidates: list[dict[str, Any]] = []
        for key, stats in bins.items():
            if key[0] not in hour_set or stats.count < _MIN_BIN_SAMPLES:
                continue
            if stats.reachable_count <= 0 and not allow_observed_only:
                continue
            if stats.reachable_count <= 0 and stats.count < _MIN_OBSERVED_ONLY_BIN_SAMPLES:
                continue
            suppress_until = int(suppressed.get(key, 0) or 0)
            if suppress_until > current_minute:
                continue
            target_lat = stats.lat
            target_lng = stats.lng
            distance_km = self._haversine_km(current_lat, current_lng, target_lat, target_lng)
            if distance_km < _MIN_REPOSITION_KM or distance_km > _MAX_REPOSITION_KM:
                continue
            age_minutes = max(0, current_minute - stats.last_seen_minute)
            stale_penalty = min(_MAX_STALE_PENALTY, age_minutes / 60.0 * _STALE_PENALTY_PER_HOUR)
            density_bonus = min(stats.profitable_count, _MAX_PROFITABLE_WEIGHT_COUNT) * _PROFITABLE_COUNT_WEIGHT
            reposition_cost = distance_km * _COST_PER_KM
            confidence = 1.0 if stats.reachable_count >= _MIN_BIN_SAMPLES else _OBSERVED_ONLY_CONFIDENCE
            target_score = (stats.top_avg_net + density_bonus) * confidence - reposition_cost - stale_penalty
            candidates.append(
                {
                    "hotspot_key": key,
                    "hour": key[0],
                    "latitude": target_lat,
                    "longitude": target_lng,
                    "hotspot_sample_count": stats.count,
                    "reachable_sample_count": stats.reachable_count,
                    "top_avg_net": stats.top_avg_net,
                    "best_net": stats.best_net,
                    "avg_pickup_km": stats.avg_pickup_km,
                    "confidence": confidence,
                    "reposition_km": distance_km,
                    "reposition_cost": reposition_cost,
                    "stale_penalty": stale_penalty,
                    "target_score": target_score,
                }
            )
        return candidates

    def _update_recent_reposition_feedback(
        self,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        has_reachable_items: bool,
    ) -> None:
        last = self._last_reposition_by_driver.get(driver_id)
        if not last:
            return
        last_minute = int(last.get("minute", 0) or 0)
        if current_minute <= last_minute:
            return
        key = last.get("hotspot_key")
        lat = self._safe_float(last.get("latitude"))
        lng = self._safe_float(last.get("longitude"))
        if not isinstance(key, tuple) or lat is None or lng is None:
            self._last_reposition_by_driver.pop(driver_id, None)
            return
        if self._haversine_km(current_lat, current_lng, float(lat), float(lng)) > _ARRIVAL_RADIUS_KM:
            return
        if has_reachable_items:
            self._last_reposition_by_driver.pop(driver_id, None)
            return
        suppressed = self._suppressed_until_by_driver.setdefault(driver_id, {})
        suppressed[key] = max(int(suppressed.get(key, 0) or 0), current_minute + _DEBOUNCE_MINUTES)
        self._last_reposition_by_driver.pop(driver_id, None)

    def _is_low_market_for_driver(self, driver_id: str, sample: dict[str, Any]) -> bool:
        history = self._market_history_by_driver.get(driver_id, [])
        if len(history) < _MARKET_HISTORY_MIN_SAMPLES:
            return False
        current_score = self._safe_float(sample.get("score"))
        if current_score is None:
            return False
        return float(current_score) <= self._percentile(history, _MARKET_LOW_QUANTILE)

    def _trim_old_bins(self, driver_id: str) -> None:
        bins = self._bins_by_driver.get(driver_id)
        if not bins or len(bins) <= _MAX_BINS_PER_DRIVER:
            return
        ordered = sorted(bins.items(), key=lambda item: item[1].last_seen_minute)
        for key, _stats in ordered[: len(bins) - _MAX_BINS_PER_DRIVER]:
            bins.pop(key, None)

    def sample_count(self, driver_id: str) -> int:
        return self._total_samples_by_driver.get(driver_id, 0)

    def observation_count(self, driver_id: str) -> int:
        return self._observation_count_by_driver.get(driver_id, 0)

    def estimate_location_value(self, driver_id: str, current_minute: int, lat: float, lng: float) -> float:
        if self._total_samples_by_driver.get(driver_id, 0) < _MIN_DRIVER_SAMPLES:
            return 0.0
        current_hour = self._hour_of_day(current_minute)
        hour_set = {current_hour, (current_hour + 1) % 24, (current_hour + 2) % 24, (current_hour + 3) % 24}
        best = 0.0
        for key, stats in self._bins_by_driver.get(driver_id, {}).items():
            if key[0] not in hour_set or stats.count < _MIN_BIN_SAMPLES:
                continue
            distance_km = self._haversine_km(lat, lng, stats.lat, stats.lng)
            if distance_km > _DESTINATION_VALUE_RADIUS_KM:
                continue
            age_minutes = max(0, current_minute - stats.last_seen_minute)
            stale_penalty = min(_MAX_STALE_PENALTY, age_minutes / 60.0 * _STALE_PENALTY_PER_HOUR)
            density_bonus = min(stats.profitable_count, _MAX_PROFITABLE_WEIGHT_COUNT) * _PROFITABLE_COUNT_WEIGHT
            confidence = 1.0 if stats.reachable_count >= _MIN_BIN_SAMPLES else _OBSERVED_ONLY_CONFIDENCE
            value = (stats.top_avg_net + density_bonus) * confidence - distance_km * _COST_PER_KM - stale_penalty
            best = max(best, value)
        return min(_DESTINATION_VALUE_MAX, max(0.0, best))

    def _can_use_observed_only(self, driver_id: str) -> bool:
        return self._observation_count_by_driver.get(driver_id, 0) >= _MIN_OBSERVED_ONLY_OBSERVATIONS

    @staticmethod
    def _hotspot_key(hour: int, lat: float, lng: float) -> HotspotKey:
        return (
            int(hour) % 24,
            int(math.floor(lat / _GRID_SIZE_DEGREES)),
            int(math.floor(lng / _GRID_SIZE_DEGREES)),
        )

    @staticmethod
    def _hour_of_day(current_minute: int) -> int:
        return (int(current_minute) % 1440) // 60

    def _estimate_local_order_net(self, cargo: dict[str, Any]) -> float:
        price = self._safe_float(cargo.get("price")) or 0.0
        haul_km = self._cargo_haul_distance_km(cargo)
        return price - haul_km * _COST_PER_KM

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

    @classmethod
    def _cargo_id(cls, item: dict[str, Any]) -> str:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        return str(cargo.get("cargo_id", "") or "").strip()

    def _cargo_observation_hour(self, cargo: dict[str, Any], current_minute: int) -> int:
        load_time = cargo.get("load_time")
        if isinstance(load_time, list) and load_time:
            minute = self._parse_wall_time_minutes(load_time[0])
            if minute is not None:
                return self._hour_of_day(minute)
        minute = self._parse_wall_time_minutes(cargo.get("create_time"))
        if minute is not None:
            return self._hour_of_day(minute)
        return self._hour_of_day(current_minute)

    @staticmethod
    def _parse_wall_time_minutes(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if len(text) < 16:
            return None
        try:
            day = int(text[8:10]) - 1
            hour = int(text[11:13])
            minute = int(text[14:16])
        except (ValueError, IndexError):
            return None
        return day * 1440 + hour * 60 + minute

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        radius = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
        return 2.0 * radius * math.asin(math.sqrt(a))

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(math.floor((len(ordered) - 1) * quantile))))
        return ordered[idx]
