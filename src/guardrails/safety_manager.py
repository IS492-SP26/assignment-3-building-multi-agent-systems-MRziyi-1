"""
Safety Manager
Coordinates input/output safety guardrails and logs all safety events.

Policy categories (from config.yaml):
- harmful_content:   violence, weapons, illegal activity, self-harm
- personal_attacks:  harassment, hate speech
- security:          prompt injection / jailbreak attempts
- misinformation:    requests to generate/spread false information
- off_topic_queries: queries unrelated to HCI research (soft, non-blocking)
"""

from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Coordinates safety guardrails for the multi-agent system.

    Usage in orchestrator:
        safety = SafetyManager(config["safety"])

        # Before processing
        check = safety.check_input_safety(query)
        if not check["safe"]:
            return early_refusal(check["message"])

        # After processing
        check = safety.check_output_safety(response, sources)
        final_response = check["response"]  # may be sanitized/blocked
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        self.safety_events: List[Dict[str, Any]] = []

        self.prohibited_categories: List[str] = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries",
        ])

        self.on_violation: Dict[str, Any] = config.get("on_violation", {})

        # Instantiate guardrails
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Validate user query before it reaches the agents.

        Returns:
            safe (bool): True if query may proceed
            violations (list): detected violations
            sanitized_input (str): cleaned query (same as input if no change)
            action (str): "refuse" | "sanitize" (only if not safe)
            message (str): refusal message shown to user (only if not safe)
        """
        if not self.enabled:
            return {"safe": True, "violations": [], "sanitized_input": query}

        validation = self.input_guardrail.validate(query)
        raw_violations = validation.get("violations", [])

        # Collect violations that are blocking (high/medium) or in prohibited categories
        safety_violations = []
        for v in raw_violations:
            category = v.get("category", v.get("validator", "unknown"))
            severity = v.get("severity", "low")
            if severity in ("high", "medium") or category in self.prohibited_categories:
                safety_violations.append({
                    "category": category,
                    "reason": v.get("reason", "Policy violation"),
                    "severity": severity,
                })

        is_safe = validation.get("valid", True)

        if not is_safe and self.log_events:
            self._log_safety_event("input", query, safety_violations, is_safe)

        result: Dict[str, Any] = {
            "safe": is_safe,
            "violations": safety_violations,
            "sanitized_input": validation.get("sanitized_input", query),
        }

        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            result["action"] = action
            result["message"] = self.on_violation.get(
                "message",
                "I cannot process this request due to safety policies.",
            )

        return result

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate agent response before it is shown to the user.

        Returns:
            safe (bool): True if response passes all checks
            violations (list): detected violations
            response (str): original or sanitized/blocked response
            action (str): "refuse" | "sanitize" (only if not safe)
        """
        if not self.enabled:
            return {"safe": True, "violations": [], "response": response}

        validation = self.output_guardrail.validate(response, sources)
        raw_violations = validation.get("violations", [])

        safety_violations = [
            {
                "category": v.get("validator", "output_safety"),
                "reason": v.get("reason", "Output safety violation"),
                "severity": v.get("severity", "medium"),
            }
            for v in raw_violations
        ]

        is_safe = validation.get("valid", True)

        if not is_safe and self.log_events:
            self._log_safety_event("output", response[:200], safety_violations, is_safe)

        # The OutputGuardrail._sanitize() already handles PII redaction vs harmful block.
        # Use its sanitized output directly; only hard-refuse for truly harmful content.
        # "Harmful" here matches output_guardrail._sanitize's blocking criteria:
        # high-severity, non-PII violations. Low-severity findings (e.g.
        # citation_consistency) must NOT trigger a wholesale refusal.
        has_harmful = any(
            v.get("severity") == "high" and v.get("validator") != "pii"
            for v in raw_violations
        )

        result: Dict[str, Any] = {
            "safe": is_safe,
            "violations": safety_violations,
            "response": validation.get("sanitized_output", response),
        }

        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            result["action"] = action
            if action == "refuse" and has_harmful:
                result["response"] = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies.",
                )
            # For PII-only violations: use the redacted sanitized_output (already set above)

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """Delegate sanitization to OutputGuardrail."""
        return self.output_guardrail._sanitize(response, violations)

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
    ) -> None:
        """Record a safety event in memory and optionally to file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": (
                content[:100] + "..." if len(content) > 100 else content
            ),
        }

        self.safety_events.append(event)

        summary = "; ".join(v.get("reason", "") for v in violations[:3])
        self.logger.warning(
            f"Safety event [{event_type}] safe={is_safe}: {summary}"
        )

        log_file = self.config.get("safety_log_file")
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    # ── Stats / inspection ─────────────────────────────────────────────────────

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Return all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across all safety events."""
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0.0,
        }

    def clear_events(self) -> None:
        """Clear in-memory safety event log."""
        self.safety_events = []
