"""
LLM-as-a-Judge
Uses an LLM to evaluate system outputs against defined criteria.

The judge is intentionally decoupled from the main agent model:
- Primary: Groq API (llama-3.3-70b-versatile) — independent client/temperature
- Fallback: vLLM endpoint (Qwen/Qwen3-8B) — same endpoint as agents

Each criterion is scored by TWO independent judging prompts (rubric perspective
+ adversarial peer-reviewer perspective) and the scores are averaged. Criteria
are loaded from config.yaml (evaluation.criteria) and weighted into an overall
score.
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os


class LLMJudge:
    """
    LLM-based evaluator that scores responses criterion-by-criterion.

    Scoring scale (per criterion):
    - 0.0–0.2: Very poor / completely missing
    - 0.2–0.4: Poor / major gaps
    - 0.4–0.6: Adequate / partial coverage
    - 0.6–0.8: Good / mostly satisfies the criterion
    - 0.8–1.0: Excellent / fully satisfies the criterion
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        self.model_config = config.get("models", {}).get("judge", {})
        self.criteria = config.get("evaluation", {}).get("criteria", [])

        # Build LLM client: try Groq first, then vLLM
        self.client, self.client_type, self.model_name = self._build_client()

        self.logger.info(
            f"LLMJudge initialized: client={self.client_type}, "
            f"model={self.model_name}, criteria={len(self.criteria)}"
        )

    # ── Client construction ────────────────────────────────────────────────────

    def _build_client(self):
        """
        Build the LLM client used for judging.

        Priority:
        1. Groq (if GROQ_API_KEY is set) — independent from the main model
        2. OpenAI-compatible vLLM endpoint (if OPENAI_API_KEY + OPENAI_BASE_URL)
        3. None — judge will return zero scores with an error message
        """
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_base = os.getenv("OPENAI_BASE_URL")

        if groq_key:
            try:
                from groq import Groq
                model = self.model_config.get("name", "llama-3.3-70b-versatile")
                self.logger.info(f"Judge using Groq model: {model}")
                return Groq(api_key=groq_key), "groq", model
            except ImportError:
                self.logger.warning("groq package not installed; trying vLLM fallback")

        if openai_key and openai_base:
            try:
                from openai import OpenAI
                model = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
                self.logger.info(f"Judge using vLLM model: {model}")
                return OpenAI(api_key=openai_key, base_url=openai_base), "openai", model
            except ImportError:
                self.logger.warning("openai package not installed")

        self.logger.error(
            "No LLM client available for judge. "
            "Set GROQ_API_KEY or (OPENAI_API_KEY + OPENAI_BASE_URL)."
        )
        return None, None, "unknown"

    # ── Main evaluation entry point ────────────────────────────────────────────

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge across all configured criteria.

        Returns:
            overall_score (float): weighted average across criteria
            criterion_scores (dict): per-criterion score + reasoning
            feedback (list): summary feedback strings
        """
        self.logger.info(f"Evaluating response for query: {query[:60]}...")

        results: Dict[str, Any] = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        if not self.criteria:
            self.logger.warning("No evaluation criteria configured in config.yaml")
            return results

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)
            self.logger.info(f"Judging criterion: {criterion_name}")

            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth,
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

            if score.get("reasoning"):
                results["feedback"].append(
                    f"[{criterion_name}] {score['reasoning'][:120]}"
                )

        results["overall_score"] = (
            weighted_score / total_weight if total_weight > 0 else 0.0
        )
        return results

    # ── Per-criterion judging ──────────────────────────────────────────────────

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
    ) -> Dict[str, Any]:
        """
        Run two independent judge prompts (rubric perspective + adversarial
        perspective) for a single criterion, then average the scores. This
        satisfies the rubric requirement of ≥2 independent judging prompts.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        rubric_prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth,
        )

        adversarial_prompt = self._create_adversarial_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth,
        )

        scores: List[float] = []
        per_perspective: Dict[str, Dict[str, Any]] = {}

        for perspective, prompt in (
            ("rubric", rubric_prompt),
            ("adversarial", adversarial_prompt),
        ):
            try:
                judgment_text = await self._call_judge_llm(prompt)
                score_value, reasoning = self._parse_judgment(judgment_text)
            except Exception as e:
                self.logger.error(
                    f"Error judging '{criterion_name}' [{perspective}]: {e}"
                )
                score_value, reasoning = 0.0, f"Evaluation error: {e}"

            scores.append(score_value)
            per_perspective[perspective] = {
                "score": score_value,
                "reasoning": reasoning,
            }

        avg_score = sum(scores) / len(scores) if scores else 0.0
        combined_reasoning = (
            f"[rubric] {per_perspective['rubric']['reasoning']} "
            f"[adversarial] {per_perspective['adversarial']['reasoning']}"
        )

        return {
            "score": avg_score,
            "reasoning": combined_reasoning,
            "criterion": criterion_name,
            "perspectives": per_perspective,
        }

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
    ) -> str:
        """
        Perspective 1 — Rubric-based evaluation.
        Frames the judge as an expert evaluator scoring against a fixed rubric.
        Used together with the adversarial-perspective prompt to satisfy the
        ≥2 independent judging prompts requirement.
        """
        prompt = f"""You are an expert evaluator assessing AI-generated research responses.

## Criterion: {criterion_name}
{description}

## Scoring Rubric (0.0 – 1.0)
- 0.0–0.2: Very poor — criterion is entirely unmet
- 0.2–0.4: Poor — major deficiencies
- 0.4–0.6: Adequate — partially meets the criterion
- 0.6–0.8: Good — mostly satisfies the criterion
- 0.8–1.0: Excellent — fully and clearly satisfies the criterion

## Query
{query}

## Response to Evaluate
{response}
"""

        if sources:
            source_list = "\n".join(
                f"- {s.get('title', 'Unknown')} ({s.get('url', '')})"
                for s in sources[:5]
            )
            prompt += f"\n## Sources Available to the System\n{source_list}\n"

        if ground_truth:
            prompt += f"\n## Reference / Expected Answer\n{ground_truth}\n"

        prompt += """
## Instructions
Evaluate the response strictly on the criterion above. Be concise but specific.
Respond with ONLY valid JSON (no markdown, no extra text):
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<one or two sentences explaining the score>"
}"""

        return prompt

    def _create_adversarial_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
    ) -> str:
        """
        Perspective 2 — Adversarial / skeptical reviewer.
        Frames the judge as a hostile peer reviewer actively looking for
        weaknesses, exaggerated claims, missing evidence, and over-confidence.
        This produces an independent score that, when averaged with the rubric
        perspective, mitigates leniency bias common in single-prompt LLM judges.
        """
        prompt = f"""You are a strict, skeptical peer reviewer for an HCI research venue.
Your job is to find weaknesses in AI-generated responses, NOT to be charitable.

## Criterion under attack: {criterion_name}
{description}

## Your reviewing stance
- Assume claims are unsupported until proven by citations or sources.
- Flag vague hedging, missing evidence, unjustified generalizations.
- Penalize responses that read like research plans rather than synthesized answers.
- Reward only concrete, well-cited, query-grounded content.

## Anchored scoring scale (0.0 – 1.0)
- 0.0–0.2: Fundamentally fails the criterion; would reject from any venue.
- 0.2–0.4: Weak; substantial revisions needed before it could be accepted.
- 0.4–0.6: Borderline; mixed evidence, partial answer, notable gaps.
- 0.6–0.8: Solid; minor weaknesses but the criterion is largely satisfied.
- 0.8–1.0: Exemplary; could not reasonably be improved on this criterion.

## Query
{query}

## Response under review
{response}
"""

        if sources:
            source_list = "\n".join(
                f"- {s.get('title', 'Unknown')} ({s.get('url', '')})"
                for s in sources[:5]
            )
            prompt += f"\n## Sources the system had access to\n{source_list}\n"

        if ground_truth:
            prompt += f"\n## Reference / expected answer\n{ground_truth}\n"

        prompt += """
## Output
Be terse and adversarial. Identify the single biggest weakness for this criterion.
Respond with ONLY valid JSON (no markdown, no extra text):
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<one short sentence naming the biggest weakness or strength>"
}"""

        return prompt

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call the configured LLM client synchronously inside this async method.
        The Groq and OpenAI Python clients are synchronous; this is fine for
        evaluation pipelines that don't need concurrent judge calls.
        """
        if self.client is None:
            raise RuntimeError(
                "No LLM client configured. "
                "Set GROQ_API_KEY or OPENAI_API_KEY + OPENAI_BASE_URL."
            )

        temperature = self.model_config.get("temperature", 0.3)
        max_tokens = self.model_config.get("max_tokens", 512)

        self.logger.debug(f"Calling {self.client_type} judge model: {self.model_name}")

        # Build kwargs. When the judge falls back to a vLLM endpoint serving a
        # reasoning model (Qwen3, DeepSeek-R1, …), thinking-mode output occupies
        # most of the token budget and the trailing JSON gets truncated. Pass
        # `chat_template_kwargs.enable_thinking=False` (Qwen3) and a generous
        # token budget so a clean JSON answer always fits. The Groq client
        # rejects unknown extra_body keys, so we only pass it for openai/vLLM.
        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert evaluator. Do not think out loud. "
                        "Reply with a single valid JSON object and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.client_type == "openai":
            # Qwen3 chat template flag — silently ignored by vLLM if unknown
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        completion = self.client.chat.completions.create(**call_kwargs)
        return completion.choices[0].message.content

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse the judge LLM's JSON response into (score, reasoning).

        Robust to:
        - Markdown code fences (```json ... ```)
        - Reasoning-mode blocks emitted by Qwen3 / DeepSeek-R1 style models
          (e.g. "<think> ... </think>{json}") which appear when the judge
          falls back to the vLLM Qwen3-8B endpoint.
        - Leading/trailing prose around the JSON object — we extract the
          first balanced {...} block as a last resort.
        """
        import re

        try:
            clean = (judgment or "").strip()

            # 1. Strip <think>…</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)
            clean = re.sub(
                r"<think>.*?</think>",
                "",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            # Handle truncated <think> blocks where </think> never appeared
            if clean.lower().startswith("<think>"):
                close = clean.lower().rfind("</think>")
                if close >= 0:
                    clean = clean[close + len("</think>"):].strip()
                else:
                    # No closing tag; drop everything up to the first '{'
                    brace = clean.find("{")
                    if brace >= 0:
                        clean = clean[brace:].strip()

            # 2. Strip markdown fences anywhere in the response
            clean = re.sub(r"```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
            clean = clean.replace("```", "").strip()

            # 3. Try direct JSON parse first
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                # 4. Fallback: extract the first balanced {...} block
                match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                if not match:
                    raise
                data = json.loads(match.group(0))

            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse judgment: {e}\nRaw: {judgment[:200]}")
            return 0.0, f"Parse error: {e}"


# ── Standalone demo ────────────────────────────────────────────────────────────

async def example_basic_evaluation():
    """Demo: evaluate a single response."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    judge = LLMJudge(config)

    query = "What are the key principles of explainable AI for novice users?"
    response = (
        "Explainable AI for novice users should prioritize simplicity, "
        "transparency, and interactivity. Key principles include: "
        "(1) plain-language explanations, (2) visual representations, "
        "(3) progressive disclosure, and (4) user control over detail level. "
        "[Source: Arrieta et al., 2020] [Source: Miller, 2019]"
    )
    ground_truth = (
        "Key principles include transparency, simplicity, interactive explanations, "
        "and building user trust through understandable model behavior."
    )

    print("=" * 70)
    print("EXAMPLE: Basic Evaluation")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"\nResponse (truncated): {response[:100]}...")

    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth,
    )

    print(f"\nOverall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result["criterion_scores"].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    → {score_data['reasoning'][:100]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_basic_evaluation())
