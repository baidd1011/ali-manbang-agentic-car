"""LLM guard for high-risk preferences that failed compilation."""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

_NO_FEASIBLE_CARGO_WAIT_MINUTES = 15
_MAX_GUARD_CANDIDATES = 60
_GUARD_WAIT_MAX_MINUTES = 720


class UnknownPreferenceGuard:
    """Conservative guard for high-penalty unknown preference rules."""

    def __init__(self, api: SimulationApiPort, logger: logging.Logger | None = None) -> None:
        self._api = api
        self._logger = logger or logging.getLogger("agent.unknown_preference_guard")

    def has_high_risk_unknown(self, rules: list[dict[str, Any]]) -> bool:
        return bool(self._high_risk_unknown_rules(rules))

    def filter_items(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        items: list[dict[str, Any]],
        decision_time_min: int,
    ) -> list[dict[str, Any]]:
        unknown_rules = self._high_risk_unknown_rules(rules)
        if not unknown_rules or not items:
            return items
        filtered: list[dict[str, Any]] = []
        blocked_ids: set[str] = set()
        reviewed_count = 0
        for start in range(0, len(items), _MAX_GUARD_CANDIDATES):
            reviewed_items = items[start : start + _MAX_GUARD_CANDIDATES]
            reviewed_count += len(reviewed_items)
            try:
                data = self._call_guard(
                    task="filter_cargo_candidates",
                    driver_id=driver_id,
                    status=status,
                    decision_time_min=decision_time_min,
                    unknown_rules=unknown_rules,
                    cargo_candidates=[self._cargo_summary(item) for item in reviewed_items],
                )
            except Exception as exc:  # pragma: no cover - runtime safety path
                self._logger.warning("unknown preference guard filter failed driver_id=%s error=%s", driver_id, exc)
                return []
            decision = str(data.get("decision", "uncertain") or "uncertain").strip().lower()
            if decision in {"block_all", "wait", "uncertain"}:
                self._logger.info("unknown preference guard blocked all cargo driver_id=%s decision=%s", driver_id, decision)
                return []
            batch_blocked_ids = {
                str(cargo_id).strip()
                for cargo_id in data.get("blocked_cargo_ids", [])
                if str(cargo_id).strip()
            }
            blocked_ids.update(batch_blocked_ids)
            filtered.extend(
                item
                for item in reviewed_items
                if self._cargo_id(item) and self._cargo_id(item) not in batch_blocked_ids
            )
        if len(filtered) != len(items):
            self._logger.info(
                "unknown preference guard filtered cargo driver_id=%s before=%s after=%s reviewed=%s blocked=%s",
                driver_id,
                len(items),
                len(filtered),
                reviewed_count,
                len(blocked_ids),
            )
        return filtered

    def guard_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        rules: list[dict[str, Any]],
        action: dict[str, Any],
        decision_time_min: int,
        reachable_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unknown_rules = self._high_risk_unknown_rules(rules)
        if not unknown_rules or action.get("action") == "wait":
            return action
        try:
            data = self._call_guard(
                task="guard_action",
                driver_id=driver_id,
                status=status,
                decision_time_min=decision_time_min,
                unknown_rules=unknown_rules,
                proposed_action=self._action_summary(action, reachable_items),
            )
        except Exception as exc:  # pragma: no cover - runtime safety path
            self._logger.warning("unknown preference guard action failed driver_id=%s error=%s", driver_id, exc)
            return self._wait_action()
        decision = str(data.get("decision", "uncertain") or "uncertain").strip().lower()
        if decision == "allow":
            return action
        wait_minutes = self._safe_wait_minutes(data.get("wait_minutes"))
        self._logger.info(
            "unknown preference guard replaced action driver_id=%s action=%s decision=%s reason=%s wait_minutes=%s",
            driver_id,
            action.get("action"),
            decision,
            data.get("reason"),
            wait_minutes,
        )
        return {"action": "wait", "params": {"duration_minutes": wait_minutes}}

    def _call_guard(self, **payload: Any) -> dict[str, Any]:
        response = self._api.model_chat_completion(
            {
                "enable_thinking": True,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
            raise ValueError("guard response must be object")
        return data

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是高罚司机偏好兜底审查器。只输出JSON对象，禁止markdown和解释。"
            "输出格式：{\"decision\":\"allow|block_all|wait|uncertain\","
            "\"blocked_cargo_ids\":[字符串],\"wait_minutes\":正整数或null,\"reason\":\"简短原因\"}。"
            "输入中的 unknown_rules 是偏好编译失败但罚金较高的原文。"
            "filter_cargo_candidates 任务：判断候选货源是否可能违反这些原文偏好；明显违反或不确定就把 cargo_id 放入 blocked_cargo_ids。"
            "guard_action 任务：判断拟执行动作是否可能违反这些原文偏好；明显安全才 allow，不确定就 wait。"
            "只依据输入给出的状态、坐标、城市、货名、时间和偏好原文判断；不要编造缺失事实。"
        )

    @staticmethod
    def _high_risk_unknown_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "content": str(rule.get("source_content", "") or ""),
                "penalty_amount": rule.get("penalty_amount"),
                "penalty_cap": rule.get("penalty_cap"),
                "compile_error": rule.get("compile_error"),
            }
            for rule in rules
            if rule.get("rule_type") == "unknown"
            and rule.get("risk_level") == "high"
            and str(rule.get("source_content", "") or "").strip()
        ]

    @classmethod
    def _cargo_summary(cls, item: dict[str, Any]) -> dict[str, Any]:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        return {
            "cargo_id": cls._cargo_id(item),
            "cargo_name": cargo.get("cargo_name"),
            "price": cargo.get("price"),
            "start": cargo.get("start"),
            "end": cargo.get("end"),
            "distance_km": item.get("distance_km"),
            "load_time": cargo.get("load_time"),
            "arrival_minutes": item.get("arrival_minutes"),
            "finish_minutes": item.get("finish_minutes"),
        }

    @classmethod
    def _action_summary(cls, action: dict[str, Any], reachable_items: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"action": action.get("action"), "params": action.get("params")}
        if action.get("action") != "take_order":
            return summary
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        for item in reachable_items:
            if cls._cargo_id(item) == cargo_id:
                summary["cargo"] = cls._cargo_summary(item)
                break
        return summary

    @staticmethod
    def _cargo_id(item: dict[str, Any]) -> str:
        cargo = item.get("cargo") if isinstance(item.get("cargo"), dict) else {}
        return str(cargo.get("cargo_id", "") or "").strip()

    @staticmethod
    def _safe_wait_minutes(value: Any) -> int:
        try:
            minutes = int(float(value))
        except (TypeError, ValueError):
            return _NO_FEASIBLE_CARGO_WAIT_MINUTES
        return max(_NO_FEASIBLE_CARGO_WAIT_MINUTES, min(_GUARD_WAIT_MAX_MINUTES, minutes))

    @staticmethod
    def _wait_action() -> dict[str, Any]:
        return {"action": "wait", "params": {"duration_minutes": _NO_FEASIBLE_CARGO_WAIT_MINUTES}}
