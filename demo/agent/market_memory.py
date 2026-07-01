"""Per-driver market memory learned only from live cargo queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_GRID_SIZE_DEGREES = 0.1
_HISTORY_MAX_SAMPLES = 500
_RECENT_HISTORY_MINUTES = 7 * 24 * 60
_NO_FEASIBLE_CARGO_WAIT_MINUTES = 15
_QUERY_K_NORMAL = 100
_QUERY_K_EXPLORE = 600
_QUERY_EXPLORE_MAX_SCAN_MINUTES = 60
_DRIVER_BOOTSTRAP_EXPLORE_QUERIES = 3
_DRIVER_MAX_EXPLORE_QUERIES = 10
_DRIVER_DAILY_MAX_EXPLORE_QUERIES = 2
_BUCKET_EXPLORE_COOLDOWN_MINUTES = 180
_BUCKET_EXPLORE_MIN_INTERVAL_MINUTES = 360
_MIN_DRIVER_QUERY_SAMPLES = 8
_MIN_HOUR_QUERY_SAMPLES = 2
_MIN_GOOD_HOUR_SAMPLES = 2
_MAX_DYNAMIC_WAIT_MINUTES = 30
_POOR_HOUR_NO_REACHABLE_RATE = 0.75
_POOR_HOUR_MAX_AVG_REACHABLE = 0.5
_POOR_HOUR_MAX_AVG_SCORE = -1000.0
_GOOD_HOUR_MIN_AVG_REACHABLE = 1.0
_GOOD_HOUR_MIN_SCORE = 50.0
_GOOD_HOUR_MIN_SCORE_IMPROVEMENT = 200.0
_GOOD_HOUR_LOOKAHEAD_HOURS = 8


@dataclass(frozen=True)
class QuerySnapshot:
    minute: int
    hour: int
    grid: tuple[int, int]
    raw_count: int
    reachable_count: int
    profitable_count: int
    best_net: float
    score: float
    k_used: int = _QUERY_K_NORMAL
    query_scan_minutes: int = 0
    raw_profitable_count: int = 0
    raw_best_net: float = 0.0
    raw_avg_net: float = 0.0
    raw_score: float = 0.0


@dataclass
class _HourStats:
    query_count: int = 0
    no_reachable_count: int = 0
    raw_count_sum: int = 0
    reachable_count_sum: int = 0
    score_sum: float = 0.0
    best_score: float = float("-inf")
    raw_score_sum: float = 0.0
    query_scan_minutes_sum: int = 0

    def add(self, snapshot: QuerySnapshot) -> None:
        self.query_count += 1
        self.no_reachable_count += 1 if snapshot.reachable_count <= 0 else 0
        self.raw_count_sum += snapshot.raw_count
        self.reachable_count_sum += snapshot.reachable_count
        self.score_sum += snapshot.score
        self.raw_score_sum += snapshot.raw_score
        self.query_scan_minutes_sum += snapshot.query_scan_minutes
        self.best_score = max(self.best_score, snapshot.score)

    @property
    def avg_reachable_count(self) -> float:
        return self.reachable_count_sum / self.query_count if self.query_count else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.query_count if self.query_count else 0.0

    @property
    def avg_raw_score(self) -> float:
        return self.raw_score_sum / self.query_count if self.query_count else 0.0

    @property
    def avg_query_scan_minutes(self) -> float:
        return self.query_scan_minutes_sum / self.query_count if self.query_count else 0.0

    @property
    def no_reachable_rate(self) -> float:
        return self.no_reachable_count / self.query_count if self.query_count else 0.0

    @property
    def is_poor(self) -> bool:
        return (
            self.query_count >= _MIN_HOUR_QUERY_SAMPLES
            and self.no_reachable_rate >= _POOR_HOUR_NO_REACHABLE_RATE
            and self.avg_reachable_count <= _POOR_HOUR_MAX_AVG_REACHABLE
            and self.avg_score <= _POOR_HOUR_MAX_AVG_SCORE
        )

    @property
    def is_good(self) -> bool:
        return (
            self.query_count >= _MIN_GOOD_HOUR_SAMPLES
            and self.avg_reachable_count >= _GOOD_HOUR_MIN_AVG_REACHABLE
            and self.avg_score >= _GOOD_HOUR_MIN_SCORE
        )


class MarketMemory:
    """Track market metadata per driver without reading future cargo data."""

    def __init__(self) -> None:
        self._history_by_driver: dict[str, list[QuerySnapshot]] = {}

    def remember_query(
        self,
        *,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        raw_items: Any,
        reachable_items: list[dict[str, Any]],
        market_sample: dict[str, Any],
        k_used: int = _QUERY_K_NORMAL,
        query_scan_minutes: int = 0,
    ) -> None:
        raw_nets = sorted(self._estimate_item_net(item) for item in raw_items if isinstance(item, dict)) if isinstance(raw_items, list) else []
        raw_profitable_count = sum(1 for net in raw_nets if net > 0)
        raw_best_net = raw_nets[-1] if raw_nets else 0.0
        raw_avg_net = sum(raw_nets) / len(raw_nets) if raw_nets else 0.0
        raw_top = raw_nets[-5:]
        raw_score = (sum(raw_top) / len(raw_top) + raw_profitable_count * 5.0) if raw_top else -10000.0
        snapshot = QuerySnapshot(
            minute=int(current_minute),
            hour=self._hour_of_day(current_minute),
            grid=self._grid_key(current_lat, current_lng),
            raw_count=len(raw_items) if isinstance(raw_items, list) else 0,
            reachable_count=len(reachable_items),
            profitable_count=int(market_sample.get("profitable_count", 0) or 0),
            best_net=self._safe_float(market_sample.get("best_net")) or 0.0,
            score=self._safe_float(market_sample.get("score")) or 0.0,
            k_used=max(1, int(k_used)),
            query_scan_minutes=max(0, int(query_scan_minutes)),
            raw_profitable_count=raw_profitable_count,
            raw_best_net=raw_best_net,
            raw_avg_net=raw_avg_net,
            raw_score=raw_score,
        )
        history = self._history_by_driver.setdefault(driver_id, [])
        history.append(snapshot)
        if len(history) > _HISTORY_MAX_SAMPLES:
            del history[: len(history) - _HISTORY_MAX_SAMPLES]

    def choose_query_k(
        self,
        *,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        max_query_minutes: int | None = None,
    ) -> dict[str, Any]:
        safe_minutes = _QUERY_EXPLORE_MAX_SCAN_MINUTES if max_query_minutes is None else int(max_query_minutes)
        if safe_minutes < _QUERY_EXPLORE_MAX_SCAN_MINUTES:
            return {"k": _QUERY_K_NORMAL, "reason": "preference_time_cap", "safe_minutes": safe_minutes}

        full_history = self._full_history(driver_id, current_minute)
        history = self._recent_history(driver_id, current_minute)
        grid = self._grid_key(current_lat, current_lng)
        hour = self._hour_of_day(current_minute)
        if not self._can_use_explore_query(full_history, current_minute, grid, hour):
            return {"k": _QUERY_K_NORMAL, "reason": "explore_budget_or_cooldown", "safe_minutes": safe_minutes}

        if self._explore_query_count(full_history) < _DRIVER_BOOTSTRAP_EXPLORE_QUERIES:
            return {"k": _QUERY_K_EXPLORE, "reason": "driver_cold_start", "safe_minutes": safe_minutes}

        bucket_history = [snapshot for snapshot in history if snapshot.grid == grid and snapshot.hour == hour]
        if not bucket_history:
            return {"k": _QUERY_K_NORMAL, "reason": "new_grid_hour_probe", "safe_minutes": safe_minutes}

        last_bucket = max(bucket_history, key=lambda snapshot: snapshot.minute)
        if (
            last_bucket.k_used <= _QUERY_K_NORMAL
            and last_bucket.raw_count >= last_bucket.k_used
            and last_bucket.reachable_count <= 0
        ):
            return {"k": _QUERY_K_EXPLORE, "reason": "normal_query_truncated_no_reachable", "safe_minutes": safe_minutes}
        return {"k": _QUERY_K_NORMAL, "reason": "enough_market_memory", "safe_minutes": safe_minutes}

    def market_state(
        self,
        *,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
    ) -> dict[str, Any]:
        history = self._recent_history(driver_id, current_minute)
        grid = self._grid_key(current_lat, current_lng)
        hour = self._hour_of_day(current_minute)
        stats = self._stats_by_bucket(history).get((grid, hour))
        if len(history) < _MIN_DRIVER_QUERY_SAMPLES:
            state = "cold_start"
        elif stats is None:
            state = "unseen_grid_hour"
        elif stats.is_poor:
            state = "low"
        elif stats.is_good:
            state = "good"
        else:
            state = "normal"
        return {
            "state": state,
            "sample_count": len(history),
            "grid": grid,
            "hour": hour,
            "query_count": stats.query_count if stats else 0,
            "avg_reachable_count": stats.avg_reachable_count if stats else 0.0,
            "avg_score": stats.avg_score if stats else 0.0,
            "avg_raw_score": stats.avg_raw_score if stats else 0.0,
            "avg_query_scan_minutes": stats.avg_query_scan_minutes if stats else 0.0,
            "no_reachable_rate": stats.no_reachable_rate if stats else 0.0,
        }

    def suggest_no_reachable_wait(
        self,
        *,
        driver_id: str,
        current_minute: int,
        current_lat: float,
        current_lng: float,
        market_sample: dict[str, Any],
    ) -> dict[str, Any]:
        current_reachable = int(market_sample.get("reachable_count", 0) or 0)
        if current_reachable > 0:
            return self._wait_plan(_NO_FEASIBLE_CARGO_WAIT_MINUTES, "has_reachable_items", driver_id)

        history = self._recent_history(driver_id, current_minute)
        if len(history) < _MIN_DRIVER_QUERY_SAMPLES:
            return self._wait_plan(_NO_FEASIBLE_CARGO_WAIT_MINUTES, "cold_start", driver_id, sample_count=len(history))

        current_hour = self._hour_of_day(current_minute)
        current_grid = self._grid_key(current_lat, current_lng)
        bucket_stats = self._stats_by_bucket(history)
        current_stats = bucket_stats.get((current_grid, current_hour))
        if current_stats is None or not current_stats.is_poor:
            return self._wait_plan(
                _NO_FEASIBLE_CARGO_WAIT_MINUTES,
                "insufficient_poor_market_evidence",
                driver_id,
                sample_count=len(history),
            )

        good_hour_wait = self._minutes_to_next_good_hour(current_minute, current_grid, current_stats, bucket_stats)
        if good_hour_wait is not None:
            return self._wait_plan(
                self._bounded_wait(good_hour_wait),
                "wait_for_historically_good_hour",
                driver_id,
                sample_count=len(history),
            )

        return self._wait_plan(
            _NO_FEASIBLE_CARGO_WAIT_MINUTES,
            "no_recent_better_market",
            driver_id,
            sample_count=len(history),
        )

    def _minutes_to_next_good_hour(
        self,
        current_minute: int,
        current_grid: tuple[int, int],
        current_stats: _HourStats,
        bucket_stats: dict[tuple[tuple[int, int], int], _HourStats],
    ) -> int | None:
        minute_in_hour = int(current_minute) % 60
        for hour_offset in range(1, _GOOD_HOUR_LOOKAHEAD_HOURS + 1):
            hour = (self._hour_of_day(current_minute) + hour_offset) % 24
            stats = bucket_stats.get((current_grid, hour))
            if stats is None or not stats.is_good:
                continue
            if stats.avg_score - current_stats.avg_score < _GOOD_HOUR_MIN_SCORE_IMPROVEMENT:
                continue
            return hour_offset * 60 - minute_in_hour
        return None

    def _recent_history(self, driver_id: str, current_minute: int) -> list[QuerySnapshot]:
        lower_bound = int(current_minute) - _RECENT_HISTORY_MINUTES
        return [
            snapshot
            for snapshot in self._history_by_driver.get(driver_id, [])
            if lower_bound <= snapshot.minute <= int(current_minute)
        ]

    def _full_history(self, driver_id: str, current_minute: int) -> list[QuerySnapshot]:
        return [
            snapshot
            for snapshot in self._history_by_driver.get(driver_id, [])
            if snapshot.minute <= int(current_minute)
        ]

    def _can_use_explore_query(
        self,
        history: list[QuerySnapshot],
        current_minute: int,
        grid: tuple[int, int],
        hour: int,
    ) -> bool:
        if self._explore_query_count(history) >= _DRIVER_MAX_EXPLORE_QUERIES:
            return False
        day = int(current_minute) // 1440
        daily_explore = sum(1 for snapshot in history if snapshot.k_used >= _QUERY_K_EXPLORE and snapshot.minute // 1440 == day)
        if daily_explore >= _DRIVER_DAILY_MAX_EXPLORE_QUERIES:
            return False
        bucket_explore = [
            snapshot
            for snapshot in history
            if snapshot.grid == grid and snapshot.hour == hour and snapshot.k_used >= _QUERY_K_EXPLORE
        ]
        if not bucket_explore:
            return True
        last_explore = max(bucket_explore, key=lambda snapshot: snapshot.minute)
        if current_minute - last_explore.minute < _BUCKET_EXPLORE_MIN_INTERVAL_MINUTES:
            return False
        if (
            last_explore.reachable_count <= 0
            or last_explore.score < _GOOD_HOUR_MIN_SCORE
        ) and current_minute - last_explore.minute < _BUCKET_EXPLORE_COOLDOWN_MINUTES:
            return False
        return True

    @staticmethod
    def _explore_query_count(history: list[QuerySnapshot]) -> int:
        return sum(1 for snapshot in history if snapshot.k_used >= _QUERY_K_EXPLORE)

    @staticmethod
    def _stats_by_bucket(history: list[QuerySnapshot]) -> dict[tuple[tuple[int, int], int], _HourStats]:
        stats_by_bucket: dict[tuple[tuple[int, int], int], _HourStats] = {}
        for snapshot in history:
            stats_by_bucket.setdefault((snapshot.grid, snapshot.hour), _HourStats()).add(snapshot)
        return stats_by_bucket

    @staticmethod
    def _wait_plan(duration_minutes: int, reason: str, driver_id: str, **extra: Any) -> dict[str, Any]:
        return {
            "duration_minutes": max(_NO_FEASIBLE_CARGO_WAIT_MINUTES, int(duration_minutes)),
            "reason": reason,
            "driver_id": driver_id,
            **extra,
        }

    @staticmethod
    def _bounded_wait(duration_minutes: int) -> int:
        return max(_NO_FEASIBLE_CARGO_WAIT_MINUTES, min(_MAX_DYNAMIC_WAIT_MINUTES, int(duration_minutes)))

    @staticmethod
    def _hour_of_day(current_minute: int) -> int:
        return (int(current_minute) % 1440) // 60

    @staticmethod
    def _grid_key(lat: float, lng: float) -> tuple[int, int]:
        return (
            int(math.floor(float(lat) / _GRID_SIZE_DEGREES)),
            int(math.floor(float(lng) / _GRID_SIZE_DEGREES)),
        )

    def _estimate_item_net(self, item: dict[str, Any]) -> float:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        price = self._safe_float(cargo.get("price")) or 0.0
        pickup_km = self._safe_float(item.get("distance_km")) or 0.0
        haul_km = self._cargo_haul_distance_km(cargo)
        return price - (pickup_km + haul_km) * 1.5

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
