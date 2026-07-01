"""Deterministic ranking for already reachable cargo candidates."""

from __future__ import annotations

import math
from typing import Any, Callable

_COST_PER_KM = 1.5
_REPOSITION_SPEED_KM_PER_HOUR = 60.0
_DIRECT_TAKE_MIN_NET = 120.0
_DIRECT_TAKE_MIN_NET_GAP = 350.0
_DIRECT_TAKE_MIN_SCORE_GAP = 250.0

DestinationScoreFn = Callable[[float, float], float]


class CandidateRanker:
    """Rank reachable cargo without using future data."""

    def rank_items(
        self,
        items: list[dict[str, Any]],
        *,
        current_minute: int,
        destination_score_fn: DestinationScoreFn | None = None,
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for item in items:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            cargo_id = str(cargo.get("cargo_id", "") or "").strip()
            if not cargo_id:
                continue
            metrics = self._metrics(
                item,
                current_minute=current_minute,
                destination_score_fn=destination_score_fn,
            )
            enriched = dict(item)
            enriched["rank_metrics"] = metrics
            ranked.append(enriched)
        ranked.sort(key=lambda row: float(row.get("rank_metrics", {}).get("rank_score", 0.0)), reverse=True)
        return ranked

    def deterministic_take_action(self, ranked_items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not ranked_items:
            return None
        top = ranked_items[0]
        top_metrics = top.get("rank_metrics") if isinstance(top.get("rank_metrics"), dict) else {}
        top_net = float(top_metrics.get("estimated_net", 0.0) or 0.0)
        if top_net < _DIRECT_TAKE_MIN_NET:
            return None

        if len(ranked_items) == 1:
            return self._take_action(top)

        second_metrics = ranked_items[1].get("rank_metrics") if isinstance(ranked_items[1].get("rank_metrics"), dict) else {}
        second_net = float(second_metrics.get("estimated_net", 0.0) or 0.0)
        top_score = float(top_metrics.get("rank_score", 0.0) or 0.0)
        second_score = float(second_metrics.get("rank_score", 0.0) or 0.0)
        net_gap = top_net - second_net
        score_gap = top_score - second_score
        if net_gap >= _DIRECT_TAKE_MIN_NET_GAP or (
            net_gap >= _DIRECT_TAKE_MIN_NET and score_gap >= _DIRECT_TAKE_MIN_SCORE_GAP
        ):
            return self._take_action(top)
        return None

    def candidate_summary(self, ranked_items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in ranked_items[: max(0, int(limit))]:
            cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
            metrics = item.get("rank_metrics") if isinstance(item.get("rank_metrics"), dict) else {}
            out.append(
                {
                    "cargo_id": cargo.get("cargo_id"),
                    "estimated_net": self._round(metrics.get("estimated_net")),
                    "rank_score": self._round(metrics.get("rank_score")),
                    "net_per_hour": self._round(metrics.get("net_per_hour")),
                    "destination_score": self._round(metrics.get("destination_score")),
                    "pickup_km": self._round(metrics.get("pickup_km")),
                }
            )
        return out

    def _metrics(
        self,
        item: dict[str, Any],
        *,
        current_minute: int,
        destination_score_fn: DestinationScoreFn | None,
    ) -> dict[str, Any]:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        price = self._safe_float(cargo.get("price")) or 0.0
        pickup_km = self._safe_float(item.get("distance_km")) or 0.0
        haul_km = self._cargo_haul_distance_km(cargo)
        estimated_net = price - (pickup_km + haul_km) * _COST_PER_KM
        duration_minutes = self._duration_minutes(item, current_minute)
        net_per_hour = estimated_net / max(duration_minutes / 60.0, 1.0 / 60.0)
        destination_score = self._destination_score(cargo, destination_score_fn)
        deadline_slack = self._safe_float(item.get("minutes_until_load_deadline"))
        slack_bonus = min(max(deadline_slack or 0.0, 0.0), 240.0) * 0.05
        rank_score = (
            estimated_net
            + net_per_hour * 0.18
            + destination_score * 0.25
            - pickup_km * 0.60
            + slack_bonus
        )
        return {
            "estimated_net": estimated_net,
            "net_per_hour": net_per_hour,
            "pickup_km": pickup_km,
            "haul_km": haul_km,
            "duration_minutes": duration_minutes,
            "destination_score": destination_score,
            "minutes_until_load_deadline": deadline_slack,
            "rank_score": rank_score,
        }

    def _duration_minutes(self, item: dict[str, Any], current_minute: int) -> int:
        finish_minutes = self._safe_float(item.get("finish_minutes"))
        if finish_minutes is not None:
            return max(1, int(math.ceil(finish_minutes - current_minute)))
        pickup_minutes = self._safe_float(item.get("pickup_minutes"))
        if pickup_minutes is None:
            pickup_minutes = self._pickup_minutes(self._safe_float(item.get("distance_km")) or 0.0)
        wait_minutes = self._safe_float(item.get("wait_before_loading_minutes")) or 0.0
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        cost_time_minutes = self._safe_float(cargo.get("cost_time_minutes")) or 0.0
        return max(1, int(math.ceil(pickup_minutes + wait_minutes + cost_time_minutes)))

    def _destination_score(self, cargo: dict[str, Any], destination_score_fn: DestinationScoreFn | None) -> float:
        if destination_score_fn is None:
            return 0.0
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        lat = self._safe_float(end.get("lat"))
        lng = self._safe_float(end.get("lng"))
        if lat is None or lng is None:
            return 0.0
        try:
            return max(0.0, float(destination_score_fn(float(lat), float(lng))))
        except Exception:
            return 0.0

    @classmethod
    def _take_action(cls, item: dict[str, Any]) -> dict[str, Any] | None:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        cargo_id = str(cargo.get("cargo_id", "") or "").strip()
        if not cargo_id:
            return None
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}

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
    def _pickup_minutes(distance_km: float) -> int:
        if distance_km <= 1e-6:
            return 0
        return max(1, int(math.ceil((distance_km / _REPOSITION_SPEED_KM_PER_HOUR) * 60.0)))

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round(value: Any) -> float:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        radius = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lng2 - lng1)
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
        return 2.0 * radius * math.asin(math.sqrt(a))
