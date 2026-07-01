"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any

from agent.candidate_ranker import CandidateRanker
from agent.hotspot_planner import HotspotPlanner
from agent.market_memory import MarketMemory
from agent.preference_compiler import PreferenceCompiler
from agent.preference_policies import PreferencePolicyEngine
from agent.unknown_preference_guard import UnknownPreferenceGuard
from simkit.ports import SimulationApiPort

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_REPOSITION_SPEED_KM_PER_HOUR = 60.0
_MAX_CARGO_CANDIDATES_FOR_MODEL = 20
_QUERY_CARGO_EXPLORE_K = 600
_NO_FEASIBLE_CARGO_WAIT_MINUTES = 15


class ModelDecisionService:
    """基于大模型的单步决策：拉取状态与候选货源，请求补全并解析为结构化动作。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._preference_compiler = PreferenceCompiler(api, self._logger)
        self._policies = PreferencePolicyEngine(api, self._logger)
        self._hotspots = HotspotPlanner()
        self._market_memory = MarketMemory()
        self._unknown_guard = UnknownPreferenceGuard(api, self._logger)
        self._candidate_ranker = CandidateRanker()

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        rules = self._compile_rules(driver_id, status)
        action = self._policies.pre_query_action(driver_id, status, rules)
        if action is not None:
            return self._finalize_pre_query_action(driver_id, status, rules, action)

        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        query_start_min = int(status.get("simulation_progress_minutes", 0) or 0)
        safe_query_minutes = self._policies.max_safe_idle_wait_minutes(
            driver_id=driver_id,
            status=status,
            rules=rules,
            current_minute=query_start_min,
            requested_minutes=60,
        )
        query_plan = self._market_memory.choose_query_k(
            driver_id=driver_id,
            current_minute=query_start_min,
            current_lat=lat,
            current_lng=lng,
            max_query_minutes=safe_query_minutes,
        )
        query_k = int(query_plan.get("k", 100) or 100)
        self._logger.info(
            "query cargo plan driver_id=%s k=%s reason=%s safe_minutes=%s",
            driver_id,
            query_k,
            query_plan.get("reason"),
            query_plan.get("safe_minutes"),
        )
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=query_k)
        items = cargo_resp.get("items", [])

        decision_status = self._api.get_driver_status(driver_id)
        decision_time_min = int(decision_status.get("simulation_progress_minutes", 0) or 0)
        query_scan_minutes = max(0, decision_time_min - query_start_min)
        rules = self._compile_rules(driver_id, decision_status)
        action = self._policies.pre_query_action(driver_id, decision_status, rules)
        if action is not None:
            return self._finalize_pre_query_action(driver_id, decision_status, rules, action, raw_items=items)

        reachable_items = self._filtered_reachable_items(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            raw_items=items,
            decision_time_min=decision_time_min,
        )
        self._hotspots.observe(
            driver_id=driver_id,
            current_minute=decision_time_min,
            raw_items=items,
            reachable_items=reachable_items,
        )

        market_sample = self._policies.build_market_sample(decision_time_min, reachable_items)
        decision_lat = float(decision_status["current_lat"])
        decision_lng = float(decision_status["current_lng"])
        self._market_memory.remember_query(
            driver_id=driver_id,
            current_minute=decision_time_min,
            current_lat=decision_lat,
            current_lng=decision_lng,
            raw_items=items,
            reachable_items=reachable_items,
            market_sample=market_sample,
            k_used=query_k,
            query_scan_minutes=query_scan_minutes,
        )
        month_wait_minutes = self._policies.month_day_market_wait_minutes(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            current_minute=decision_time_min,
            market_sample=market_sample,
        )
        self._policies.remember_market_sample(driver_id, market_sample)
        if month_wait_minutes > 0:
            self._hotspots.remember_market_sample(driver_id, market_sample)
            self._logger.info(
                "monthly rest required driver_id=%s wait_minutes=%s market_score=%.2f",
                driver_id,
                month_wait_minutes,
                float(market_sample.get("score", 0.0) or 0.0),
            )
            return {"action": "wait", "params": {"duration_minutes": month_wait_minutes}}

        ranked_items = self._rank_reachable_items(driver_id, decision_time_min, reachable_items)
        market_state = self._market_memory.market_state(
            driver_id=driver_id,
            current_minute=decision_time_min,
            current_lat=decision_lat,
            current_lng=decision_lng,
        )
        self._logger.info(
            "decision input driver_id=%s time_min=%s loc=(%.5f,%.5f) cargo_items=%s reachable_items=%s "
            "market_score=%.2f market_state=%s rule_types=%s top_candidates=%s",
            driver_id,
            decision_time_min,
            lat,
            lng,
            len(items) if isinstance(items, list) else 0,
            len(reachable_items),
            float(market_sample.get("score", 0.0) or 0.0),
            market_state.get("state"),
            self._policies.rule_type_counts(rules),
            self._candidate_ranker.candidate_summary(ranked_items),
        )
        self._hotspots.remember_market_sample(driver_id, market_sample)
        if not reachable_items:
            hotspot_action = self._try_hotspot_reposition(
                driver_id=driver_id,
                status=decision_status,
                rules=rules,
                decision_time_min=decision_time_min,
                reachable_items=reachable_items,
                market_sample=market_sample,
            )
            if hotspot_action is not None:
                return hotspot_action
            wait_plan = self._market_memory.suggest_no_reachable_wait(
                driver_id=driver_id,
                current_minute=decision_time_min,
                current_lat=decision_lat,
                current_lng=decision_lng,
                market_sample=market_sample,
            )
            wait_minutes = int(wait_plan.get("duration_minutes", _NO_FEASIBLE_CARGO_WAIT_MINUTES) or _NO_FEASIBLE_CARGO_WAIT_MINUTES)
            wait_minutes = self._policies.max_safe_idle_wait_minutes(
                driver_id=driver_id,
                status=decision_status,
                rules=rules,
                current_minute=decision_time_min,
                requested_minutes=wait_minutes,
            )
            self._logger.info(
                "no reachable cargo dynamic wait driver_id=%s wait_minutes=%s reason=%s sample_count=%s",
                driver_id,
                wait_minutes,
                wait_plan.get("reason"),
                wait_plan.get("sample_count"),
            )
            return {"action": "wait", "params": {"duration_minutes": wait_minutes}}

        deterministic_action = self._try_deterministic_take_order(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            decision_time_min=decision_time_min,
            ranked_items=ranked_items,
        )
        if deterministic_action is not None:
            return deterministic_action

        hotspot_action = self._try_hotspot_reposition(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            decision_time_min=decision_time_min,
            reachable_items=ranked_items,
            market_sample=market_sample,
        )
        if hotspot_action is not None:
            return hotspot_action

        prompt = self._build_prompt(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            items=ranked_items,
            original_items_count=len(items) if isinstance(items, list) else 0,
        )
        model_resp = self._api.model_chat_completion(
            {
                "enable_thinking": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是货运调度决策器。"
                            "只允许输出一个JSON对象，格式必须是"
                            '{"action":"take_order|reposition|wait","params":{...}}。'
                            "禁止输出markdown、解释或额外文本。"
                            "当action是take_order时，params必须包含cargo_id字符串；"
                            "当action是reposition时，params必须包含latitude和longitude数值；"
                            "当action是wait时，params必须包含duration_minutes正整数。"
                            "simulation_progress_minutes 为自 2026-03-01 00:00:00 起的仿真经过分钟数。"
                            "候选货源含 load_time 为装货时间窗 [开始,结束]（墙钟）；"
                            "cargo_candidates 已过滤为能在装货窗口结束前到达的货源；"
                            "take_order 只能选择 cargo_candidates 内的 cargo_id。"
                            "若接单后无法在仿真总时长内完成装货与干线，take_order 会失败（detail 含 simulation_horizon_exceeded），且不推进时间与位置。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        )
        action = self._parse_action(model_resp)
        action = self._policies.guard_action(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            action=action,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        action = self._unknown_guard.guard_action(
            driver_id=driver_id,
            status=decision_status,
            rules=rules,
            action=action,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        self._logger.info(
            "decision output driver_id=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action

    def _finalize_pre_query_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        action: dict[str, Any],
        raw_items: Any | None = None,
    ) -> dict[str, Any]:
        decision_status = status
        decision_rules = rules
        decision_time_min = int(decision_status.get("simulation_progress_minutes", 0) or 0)
        reachable_items: list[dict[str, Any]] = []

        if action.get("action") == "take_order":
            if raw_items is None:
                lat = float(status["current_lat"])
                lng = float(status["current_lng"])
                cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=_QUERY_CARGO_EXPLORE_K)
                raw_items = cargo_resp.get("items", [])
                decision_status = self._api.get_driver_status(driver_id)
                decision_time_min = int(decision_status.get("simulation_progress_minutes", 0) or 0)
                decision_rules = self._compile_rules(driver_id, decision_status)
            reachable_items = self._filtered_reachable_items(
                driver_id=driver_id,
                status=decision_status,
                rules=decision_rules,
                raw_items=raw_items,
                decision_time_min=decision_time_min,
            )
            cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
            reachable_ids = {
                str((item.get("cargo") or {}).get("cargo_id", "")).strip()
                for item in reachable_items
                if isinstance(item, dict) and isinstance(item.get("cargo"), dict)
            }
            if cargo_id not in reachable_ids:
                self._logger.info(
                    "pre-query take_order blocked before submit driver_id=%s cargo_id=%s reachable_items=%s",
                    driver_id,
                    cargo_id,
                    len(reachable_items),
                )
                return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}

        guarded = self._policies.guard_action(
            driver_id=driver_id,
            status=decision_status,
            rules=decision_rules,
            action=action,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        guarded = self._unknown_guard.guard_action(
            driver_id=driver_id,
            status=decision_status,
            rules=decision_rules,
            action=guarded,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        self._logger.info(
            "pre-query decision output driver_id=%s action=%s params=%s",
            driver_id,
            guarded.get("action"),
            guarded.get("params"),
        )
        return guarded

    def _rank_reachable_items(
        self,
        driver_id: str,
        decision_time_min: int,
        reachable_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._candidate_ranker.rank_items(
            reachable_items,
            current_minute=decision_time_min,
            destination_score_fn=lambda lat, lng: self._hotspots.estimate_location_value(
                driver_id,
                decision_time_min,
                lat,
                lng,
            ),
        )

    def _try_deterministic_take_order(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        decision_time_min: int,
        ranked_items: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        action = self._candidate_ranker.deterministic_take_action(ranked_items)
        if action is None:
            return None
        guarded = self._policies.guard_action(
            driver_id=driver_id,
            status=status,
            rules=rules,
            action=action,
            decision_time_min=decision_time_min,
            reachable_items=ranked_items,
        )
        guarded = self._unknown_guard.guard_action(
            driver_id=driver_id,
            status=status,
            rules=rules,
            action=guarded,
            decision_time_min=decision_time_min,
            reachable_items=ranked_items,
        )
        top = ranked_items[0] if ranked_items else {}
        cargo = top.get("cargo") if isinstance(top.get("cargo"), dict) else {}
        metrics = top.get("rank_metrics") if isinstance(top.get("rank_metrics"), dict) else {}
        self._logger.info(
            "deterministic candidate decision driver_id=%s proposed_cargo=%s estimated_net=%.2f "
            "rank_score=%.2f destination_score=%.2f guarded_action=%s params=%s",
            driver_id,
            cargo.get("cargo_id"),
            float(metrics.get("estimated_net", 0.0) or 0.0),
            float(metrics.get("rank_score", 0.0) or 0.0),
            float(metrics.get("destination_score", 0.0) or 0.0),
            guarded.get("action"),
            guarded.get("params"),
        )
        return guarded

    def _try_hotspot_reposition(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        decision_time_min: int,
        reachable_items: list[dict[str, Any]],
        market_sample: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._unknown_guard.has_high_risk_unknown(rules):
            self._logger.info("hotspot reposition disabled by high-risk unknown preference driver_id=%s", driver_id)
            return None
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        plan = self._hotspots.plan_reposition(
            driver_id=driver_id,
            current_minute=decision_time_min,
            current_lat=lat,
            current_lng=lng,
            current_market_sample=market_sample,
            has_reachable_items=bool(reachable_items),
        )
        if plan is None:
            self._logger.info(
                "hotspot no_action driver_id=%s sample_count=%s has_reachable=%s current_score=%.2f block_reason=no_candidate_or_not_low_market",
                driver_id,
                self._hotspots.sample_count(driver_id),
                bool(reachable_items),
                float(market_sample.get("score", 0.0) or 0.0),
            )
            return None
        action = plan.get("action")
        if not isinstance(action, dict):
            return None
        guarded = self._policies.guard_action(
            driver_id=driver_id,
            status=status,
            rules=rules,
            action=action,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        guarded = self._unknown_guard.guard_action(
            driver_id=driver_id,
            status=status,
            rules=rules,
            action=guarded,
            decision_time_min=decision_time_min,
            reachable_items=reachable_items,
        )
        meta = plan.get("meta") if isinstance(plan.get("meta"), dict) else {}
        if guarded.get("action") == "reposition":
            self._hotspots.remember_reposition(driver_id, plan, decision_time_min)
            params = guarded.get("params") if isinstance(guarded.get("params"), dict) else {}
            self._logger.info(
                "hotspot reposition driver_id=%s action_reason=%s hotspot_sample_count=%s "
                "confidence=%.2f current_score=%.2f target_score=%.2f reposition_km=%.2f target=(%.5f,%.5f)",
                driver_id,
                meta.get("action_reason"),
                meta.get("hotspot_sample_count"),
                float(meta.get("confidence", 0.0) or 0.0),
                float(meta.get("current_score", 0.0) or 0.0),
                float(meta.get("target_score", 0.0) or 0.0),
                float(meta.get("reposition_km", 0.0) or 0.0),
                float(params.get("latitude", 0.0) or 0.0),
                float(params.get("longitude", 0.0) or 0.0),
            )
            return guarded
        self._logger.info(
            "hotspot reposition blocked driver_id=%s action_reason=%s guarded_action=%s params=%s",
            driver_id,
            meta.get("action_reason"),
            guarded.get("action"),
            guarded.get("params"),
        )
        if not reachable_items:
            return guarded
        return None

    def _compile_rules(self, driver_id: str, status: dict[str, Any]) -> list[dict[str, Any]]:
        return self._preference_compiler.compile(driver_id, status.get("preferences", []))

    def _filtered_reachable_items(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        raw_items: Any,
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        reachable_items = self._filter_reachable_cargo_items(raw_items, decision_time_min)
        reachable_items = self._policies.filter_items(
            driver_id=driver_id,
            status=status,
            rules=rules,
            items=reachable_items,
            decision_time_min=decision_time_min,
        )
        return self._unknown_guard.filter_items(
            driver_id=driver_id,
            status=status,
            rules=rules,
            items=reachable_items,
            decision_time_min=decision_time_min,
        )

    def _filter_reachable_cargo_items(self, items: Any, decision_time_min: int) -> list[dict[str, Any]]:
        reachable: list[dict[str, Any]] = []
        if not isinstance(items, list):
            return reachable
        for item in items:
            if not isinstance(item, dict):
                continue
            cargo = item.get("cargo", {})
            if not isinstance(cargo, dict):
                continue
            cargo_id = str(cargo.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            remove_minutes = self._parse_optional_wall_time_minutes(cargo.get("remove_time"))
            if remove_minutes is not None and decision_time_min >= remove_minutes:
                continue
            distance_km = self._safe_float(item.get("distance_km"))
            if distance_km is None:
                continue
            pickup_minutes = self._pickup_minutes(distance_km)
            arrival_minutes = decision_time_min + pickup_minutes
            raw_load_time = cargo.get("load_time")
            load_window = self._parse_load_window_minutes(raw_load_time)
            if raw_load_time is not None and load_window is None:
                continue
            if load_window is not None and arrival_minutes > load_window[1]:
                continue
            enriched = dict(item)
            enriched["pickup_minutes"] = pickup_minutes
            enriched["arrival_minutes"] = arrival_minutes
            enriched["load_window_minutes"] = list(load_window) if load_window is not None else None
            enriched["wait_before_loading_minutes"] = (
                max(0, load_window[0] - arrival_minutes) if load_window is not None else 0
            )
            enriched["minutes_until_load_deadline"] = (
                load_window[1] - arrival_minutes if load_window is not None else None
            )
            cost_time_minutes = int(self._safe_float(cargo.get("cost_time_minutes")) or 0)
            ready_minutes = arrival_minutes + int(enriched["wait_before_loading_minutes"])
            enriched["finish_minutes"] = ready_minutes + cost_time_minutes
            reachable.append(enriched)
        return reachable

    def _build_prompt(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        items: list[dict[str, Any]],
        original_items_count: int,
    ) -> str:
        cargo_candidates: list[dict[str, Any]] = []
        for item in items[:_MAX_CARGO_CANDIDATES_FOR_MODEL]:
            cargo = item.get("cargo", {})
            metrics = item.get("rank_metrics") if isinstance(item.get("rank_metrics"), dict) else {}
            cargo_candidates.append(
                {
                    "cargo_id": cargo.get("cargo_id"),
                    "cargo_name": cargo.get("cargo_name"),
                    "price": cargo.get("price"),
                    "cost_time_minutes": cargo.get("cost_time_minutes"),
                    "load_time": cargo.get("load_time"),
                    "start": cargo.get("start"),
                    "end": cargo.get("end"),
                    "distance_km": item.get("distance_km"),
                    "pickup_minutes": item.get("pickup_minutes"),
                    "arrival_minutes": item.get("arrival_minutes"),
                    "load_window_minutes": item.get("load_window_minutes"),
                    "wait_before_loading_minutes": item.get("wait_before_loading_minutes"),
                    "minutes_until_load_deadline": item.get("minutes_until_load_deadline"),
                    "rank_metrics": {
                        "estimated_net": metrics.get("estimated_net"),
                        "net_per_hour": metrics.get("net_per_hour"),
                        "destination_score": metrics.get("destination_score"),
                        "rank_score": metrics.get("rank_score"),
                    },
                }
            )
        decision_context: dict[str, Any] = {
            "driver_id": driver_id,
            "simulation_progress_minutes": status.get("simulation_progress_minutes"),
            "cargo_filter": {
                "original_items_count": original_items_count,
                "reachable_items_count": len(items),
                "rule": "Only listed cargo_candidates are still online at decision time, have valid load_time, can be reached before load_time ends, and satisfy compiled hard preferences.",
            },
            "driver_status": {
                "current_lat": status.get("current_lat"),
                "current_lng": status.get("current_lng"),
                "truck_length": status.get("truck_length"),
                "completed_order_count": status.get("completed_order_count"),
            },
            "compiled_preference_rule_types": self._policies.rule_type_counts(rules),
            "cargo_candidates": cargo_candidates,
        }
        unknown_preferences = self._policies.unknown_preference_contents(rules)
        if unknown_preferences:
            decision_context["uncompiled_preferences"] = unknown_preferences
        return json.dumps(decision_context, ensure_ascii=False)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pickup_minutes(distance_km: float) -> int:
        if distance_km <= 1e-6:
            return 0
        return max(1, int(math.ceil((distance_km / _REPOSITION_SPEED_KM_PER_HOUR) * 60.0)))

    @staticmethod
    def _parse_optional_wall_time_minutes(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace(" ", "T"))
        except ValueError:
            return None
        return int((dt - _SIMULATION_EPOCH).total_seconds() // 60)

    def _parse_load_window_minutes(self, value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            return None
        start = self._parse_optional_wall_time_minutes(value[0])
        end = self._parse_optional_wall_time_minutes(value[1])
        if start is None or end is None or end < start:
            return None
        return start, end

    def _parse_action(self, model_resp: dict[str, Any]) -> dict[str, Any]:
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")
        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("模型返回动作不是JSON对象")
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params")
        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"模型返回未知action: {action_name}")
        if not isinstance(params, dict):
            raise ValueError("模型返回 params 必须是对象")
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少有效 cargo_id")
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}
        if action_name == "reposition":
            latitude = float(params["latitude"])
            longitude = float(params["longitude"])
            return {"action": "reposition", "params": {"latitude": latitude, "longitude": longitude}}
        duration_minutes = int(params["duration_minutes"])
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}
