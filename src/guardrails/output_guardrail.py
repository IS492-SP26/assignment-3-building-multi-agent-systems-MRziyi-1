"""
Output Guardrail
Checks system outputs for safety violations before returning to the user.

Checks performed:
- PII detection and redaction (email, phone, SSN, credit card)
- Harmful instructional content (detailed harmful instructions in output)
- Citation consistency against retrieved sources (lightweight)
"""

from typing import Dict, Any, List, Optional
import re


# ── PII regex patterns ─────────────────────────────────────────────────────────
_PII_PATTERNS: Dict[str, str] = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    # Require separators between groups to avoid matching year ranges and numbers
    "phone": r'\b(?:\+?1[-.\s])?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]){3}\d{4}\b',
}

# ── Harmful instructional output patterns ─────────────────────────────────────
_HARMFUL_OUTPUT_PATTERNS: List[str] = [
    r'step.by.step.{0,40}(hack|exploit|attack|steal|illegal)',
    r'(how\s+to|instructions?\s+(for|to)).{0,40}(make|synthesize|create).{0,40}(drug|explosive|weapon)',
    r'(detailed|complete|full)\s+(guide|instructions?|steps).{0,30}(illegal|harmful|dangerous\s+activity)',
]


class OutputGuardrail:
    """
    Validates system outputs before they reach the user.

    Returns a result dict with:
    - valid (bool): False if any violation found
    - violations (list): all violations with type and severity
    - sanitized_output (str): output after redaction/blocking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate(
        self, response: str, sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run all output checks and return aggregated result."""
        violations: List[Dict[str, Any]] = []

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))

        if sources:
            violations.extend(self._check_factual_consistency(response, sources))

        is_valid = len(violations) == 0
        sanitized = self._sanitize(response, violations) if violations else response

        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect personally identifiable information via regex."""
        violations: List[Dict[str, Any]] = []

        for pii_type, pattern in _PII_PATTERNS.items():
            raw_matches = re.findall(pattern, text)
            # re.findall returns strings or tuples depending on groups
            flat_matches = [
                m if isinstance(m, str) else "".join(m)
                for m in raw_matches
            ]
            flat_matches = [m.strip() for m in flat_matches if m.strip()]

            if flat_matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type.replace('_', ' ')}",
                    "severity": "high",
                    "matches": flat_matches[:5],  # cap for logging
                    "pattern": pattern,
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """Detect harmful instructional content in the model's output."""
        violations: List[Dict[str, Any]] = []
        text_lower = text.lower()

        for pattern in _HARMFUL_OUTPUT_PATTERNS:
            if re.search(pattern, text_lower):
                violations.append({
                    "validator": "harmful_content",
                    "reason": "Output contains potentially harmful instructions",
                    "severity": "high",
                })
                break

        return violations

    def _check_factual_consistency(
        self, response: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Lightweight citation consistency check.
        Flags [Source: Title] citations that don't match retrieved source titles.
        """
        violations: List[Dict[str, Any]] = []
        cited_titles = re.findall(r'\[Source:\s*([^\]]+)\]', response)
        source_titles_lower = {
            s.get("title", "").lower().strip() for s in sources
        }

        for cited in cited_titles:
            if cited.lower().strip() not in source_titles_lower:
                violations.append({
                    "validator": "citation_consistency",
                    "reason": f"Citation not found in retrieved sources: {cited[:60]}",
                    "severity": "low",
                })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """Placeholder for bias detection (future work)."""
        return []

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize output:
        1. If any high-severity non-PII violation → block entire response.
        2. Otherwise → redact PII in-place.
        """
        # Block entirely for high-severity non-PII violations
        blocking = [
            v for v in violations
            if v.get("severity") == "high" and v.get("validator") != "pii"
        ]
        if blocking:
            return "This response has been blocked due to safety policy violations."

        # Redact PII with regex substitution
        sanitized = text
        for violation in violations:
            if violation.get("validator") == "pii":
                pattern = violation.get("pattern", "")
                pii_type = violation.get("pii_type", "pii")
                if pattern:
                    sanitized = re.sub(
                        pattern,
                        f"[{pii_type.upper()} REDACTED]",
                        sanitized,
                    )

        return sanitized
