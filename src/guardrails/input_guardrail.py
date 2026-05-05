"""
Input Guardrail
Checks user inputs for safety violations before processing.

Enforced policy categories:
- harmful_content: violence, weapons, self-harm, illegal activity
- personal_attacks: harassment, hate speech
- security: prompt injection / jailbreak attempts
- off_topic_queries: queries unrelated to HCI research (soft warning)
"""

from typing import Dict, Any, List
import re


# ── Harmful content regex patterns ────────────────────────────────────────────
# Each tuple: (pattern, category)
_HARMFUL_PATTERNS: List[tuple] = [
    # Violence / weapons
    (r'how\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|poison|weapon|grenade)', "harmful_content"),
    (r'(kill|murder|harm|hurt|attack)\s+(yourself|someone|a\s+person|people)', "harmful_content"),
    (r'(suicide|self.harm)\s+(method|way|how|instruction|guide|step)', "harmful_content"),
    # Illegal activity — allow words between verb and target (e.g. "hack into systems")
    (r'(hack|crack|breach|compromise|intrude).{0,40}(account|system|database|server|network|password|ai\s+system)', "harmful_content"),
    (r'(steal|exfiltrate|extract|harvest).{0,40}(user\s+data|personal\s+data|credentials|database|private\s+data)', "harmful_content"),
    (r'(make|create|buy|sell)\s+(illegal|fake)\s+(id|passport|document|credential|license)', "harmful_content"),
    (r'(drug|weapon)\s+(deal|traffic|manufactur)', "harmful_content"),
    # Explicit harmful content
    (r'(child|minor|underage).{0,30}(sexual|porn|nude|naked|explicit)', "harmful_content"),
    # Personal attacks / hate speech
    (r'\b(you\s*are|ur|you\'re)\s+(stupid|idiot|moron|worthless|useless|retard)', "personal_attacks"),
    (r'(hate|kill|exterminate)\s+(all\s+)?(women|men|blacks|whites|jews|muslims|christians|gays)', "personal_attacks"),
]

# ── Prompt injection patterns (case-insensitive substring match) ───────────────
_INJECTION_PATTERNS: List[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "forget your previous",
    "forget everything above",
    "override your",
    "bypass your instructions",
    "reveal your system prompt",
    "show me your instructions",
    "what are your system instructions",
    "you are now a",
    "act as if you are",
    "pretend you are",
    "new instructions:",
    "[[override]]",
    "### instruction",
    "<|system|>",
    "jailbreak",
]

# ── HCI-relevant keywords for soft relevance check ────────────────────────────
_HCI_KEYWORDS: List[str] = [
    "hci", "human-computer", "user interface", "ui", "ux", "usability",
    "interaction design", "accessibility", "user experience", "visualization",
    "machine learning", "augmented reality", "virtual reality", "chatbot",
    "prototype", "user study", "cognitive", "ergonomic", "interface",
    "mobile app", "website", "research", "study", "paper", "survey", "review",
    "explainable", "transparency", "trust", "ethics", "privacy", "ai ",
    "artificial intelligence", "nlp", "voice interface", "touch", "gesture",
    "wearable", "ar ", "vr ", "llm", "generative", "design pattern",
    "mental model", "feedback", "learnability", "memorability", "efficiency",
    "human factor", "affective computing",
]


class InputGuardrail:
    """
    Validates user queries against safety and relevance policies.

    Returns a result dict with:
    - valid (bool): False if any blocking violation (high/medium severity)
    - violations (list): all violations found
    - sanitized_input (str): query after any cleaning (trimmed/truncated)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_length = 3
        self.max_length = 2000

    def validate(self, query: str) -> Dict[str, Any]:
        """Run all input checks and return aggregated result."""
        violations: List[Dict[str, Any]] = []
        query_stripped = query.strip()

        # ── Length checks ──────────────────────────────────────────────────────
        if len(query_stripped) < self.min_length:
            violations.append({
                "validator": "length",
                "category": "format",
                "reason": "Query too short (minimum 3 characters)",
                "severity": "low",
            })

        if len(query_stripped) > self.max_length:
            violations.append({
                "validator": "length",
                "category": "format",
                "reason": f"Query too long (maximum {self.max_length} characters)",
                "severity": "medium",
            })
            return {
                "valid": False,
                "violations": violations,
                "sanitized_input": query_stripped[: self.max_length],
            }

        # ── Content checks ─────────────────────────────────────────────────────
        violations.extend(self._check_toxic_language(query_stripped))
        violations.extend(self._check_prompt_injection(query_stripped))
        violations.extend(self._check_relevance(query_stripped))

        # Blocking = any high or medium severity violation
        blocking = [v for v in violations if v.get("severity") in ("high", "medium")]
        is_valid = len(blocking) == 0

        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_input": query_stripped,
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """Detect harmful/toxic content using regex patterns."""
        violations: List[Dict[str, Any]] = []
        text_lower = text.lower()
        seen_categories: set = set()

        for pattern, category in _HARMFUL_PATTERNS:
            if category in seen_categories:
                continue
            if re.search(pattern, text_lower):
                seen_categories.add(category)
                violations.append({
                    "validator": "toxic_language",
                    "category": category,
                    "reason": f"Content violates policy: {category.replace('_', ' ')}",
                    "severity": "high",
                })

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect prompt injection / jailbreak attempts."""
        violations: List[Dict[str, Any]] = []
        text_lower = text.lower()

        for pattern in _INJECTION_PATTERNS:
            if pattern.lower() in text_lower:
                violations.append({
                    "validator": "prompt_injection",
                    "category": "security",
                    "reason": "Potential prompt injection or jailbreak attempt detected",
                    "severity": "high",
                })
                break  # One injection violation per query is sufficient

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Soft check: warn (low severity) if query has no HCI-related keywords.
        Does not block the query — lets the agents handle off-topic gracefully.
        """
        violations: List[Dict[str, Any]] = []
        # Skip very short queries (too ambiguous to judge relevance)
        if len(query) < 20:
            return violations

        query_lower = query.lower()
        has_relevant_terms = any(kw in query_lower for kw in _HCI_KEYWORDS)

        if not has_relevant_terms:
            violations.append({
                "validator": "relevance",
                "category": "off_topic_queries",
                "reason": (
                    "Query may be outside this system's scope "
                    "(HCI / AI transparency research topics)"
                ),
                "severity": "low",
            })

        return violations
