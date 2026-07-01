"""Read-only helpers for preference compile audit output."""

from __future__ import annotations

from typing import Any


def build_compile_audit_rows(driver_id: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        rows.append(
            {
                "driver_id": driver_id,
                "rule_type": rule.get("rule_type"),
                "params": rule.get("params"),
                "compile_confidence": rule.get("compile_confidence"),
                "compile_status": rule.get("compile_status"),
                "compile_error": rule.get("compile_error"),
                "risk_level": rule.get("risk_level"),
                "penalty_amount": rule.get("penalty_amount"),
                "penalty_cap": rule.get("penalty_cap"),
                "content": rule.get("source_content"),
            }
        )
    return rows
