"""
AutoGen-Based Orchestrator

Workflow:
1. SafetyManager checks user input (blocks unsafe queries before agents run)
2. Planner  → breaks down the research query
3. Researcher → gathers evidence via web/paper search tools
4. Writer   → synthesizes findings into a cited response
5. Critic   → evaluates quality; says TERMINATE when satisfied
6. SafetyManager checks the final output (redacts/blocks if needed)
"""

import logging
import asyncio
import concurrent.futures
import re
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat,
    with safety guardrails applied before and after agent processing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Safety manager (input + output guardrails)
        safety_config = config.get("safety", {})
        self.safety_manager = SafetyManager(safety_config)
        self.logger.info("Safety manager initialized")

        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        self.logger.info("Research team created successfully")

        self.workflow_trace: List[Dict[str, Any]] = []

    # ── Source extraction (for output safety / citation consistency) ──────────

    @staticmethod
    def _extract_sources(messages: List[Dict[str, Any]], response: str) -> List[Dict[str, str]]:
        """
        Pull a structured sources list out of the agent conversation so the
        output guardrail can run citation-consistency checks against it.

        Strategy:
          - URLs from any agent message → {url, title=url}
          - [Source: Title] tokens cited in the Writer's response → {title}
        """
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        cite_pattern = re.compile(r'\[Source:\s*([^\]]+)\]')

        seen_urls: set = set()
        seen_titles: set = set()
        sources: List[Dict[str, str]] = []

        for msg in messages:
            content = msg.get("content", "") or ""
            for url in url_pattern.findall(content):
                if url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": url, "url": url})

        for cited in cite_pattern.findall(response or ""):
            title = cited.strip()
            key = title.lower()
            if title and key not in seen_titles:
                seen_titles.add(key)
                sources.append({"title": title, "url": ""})

        return sources

    # ── Public entry point ─────────────────────────────────────────────────────

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through safety checks and the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum conversation rounds for the agent team

        Returns:
            Dict with keys: query, response, conversation_history, metadata
        """
        self.logger.info(f"Processing query: {query}")

        # ── Step 1: Input safety check ─────────────────────────────────────────
        input_check = self.safety_manager.check_input_safety(query)

        if not input_check["safe"]:
            self.logger.warning(f"Input blocked by safety policy: {query[:60]}")
            return {
                "query": query,
                "response": input_check.get(
                    "message",
                    "I cannot process this request due to safety policies."
                ),
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "plan": "",
                    "research_findings": [],
                    "critique": "",
                    "safety_events": self.safety_manager.get_safety_events(),
                    "safety_blocked": True,
                    "safety_violations": input_check.get("violations", []),
                },
            }

        # ── Step 2: Run multi-agent pipeline ──────────────────────────────────
        # Always run in a fresh thread so asyncio.run() creates its own event loop,
        # avoiding conflicts with Streamlit's ScriptRunner thread or nested loops.
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    self._process_query_async(query, max_rounds)
                ).result()
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {e}",
                "conversation_history": [],
                "metadata": {
                    "error": True,
                    "safety_events": self.safety_manager.get_safety_events(),
                },
            }

        # ── Step 3: Output safety check ────────────────────────────────────────
        sources = self._extract_sources(
            result.get("conversation_history", []),
            result.get("response", ""),
        )
        result.setdefault("metadata", {})["sources"] = sources

        output_check = self.safety_manager.check_output_safety(
            result.get("response", ""),
            sources=sources,
        )

        if not output_check["safe"]:
            self.logger.warning("Output modified by safety policy")
            result["response"] = output_check["response"]
            result["metadata"]["output_sanitized"] = True
            result["metadata"]["output_violations"] = output_check.get("violations", [])

        result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()

        self.logger.info("Query processing complete")
        return result

    async def process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async version of process_query — for callers already inside an event loop
        (e.g. the batch evaluator running under asyncio.run()).  Skips the
        ThreadPoolExecutor wrapper so the team runs in the caller's loop.
        """
        self.logger.info(f"[async] Processing query: {query}")

        input_check = self.safety_manager.check_input_safety(query)
        if not input_check["safe"]:
            self.logger.warning(f"Input blocked: {query[:60]}")
            return {
                "query": query,
                "response": input_check.get(
                    "message", "I cannot process this request due to safety policies."
                ),
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "plan": "",
                    "research_findings": [],
                    "critique": "",
                    "safety_events": self.safety_manager.get_safety_events(),
                    "safety_blocked": True,
                    "safety_violations": input_check.get("violations", []),
                },
            }

        try:
            result = await self._process_query_async(query, max_rounds)
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred: {e}",
                "conversation_history": [],
                "metadata": {"error": True, "safety_events": self.safety_manager.get_safety_events()},
            }

        sources = self._extract_sources(
            result.get("conversation_history", []),
            result.get("response", ""),
        )
        result.setdefault("metadata", {})["sources"] = sources

        output_check = self.safety_manager.check_output_safety(
            result.get("response", ""),
            sources=sources,
        )
        if not output_check["safe"]:
            result["response"] = output_check["response"]
            result["metadata"]["output_sanitized"] = True
            result["metadata"]["output_violations"] = output_check.get("violations", [])

        result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()
        return result

    # ── Async implementation ───────────────────────────────────────────────────

    async def _process_query_async(
        self, query: str, max_rounds: int = 20
    ) -> Dict[str, Any]:
        """Run the AutoGen team asynchronously and collect results."""
        task_message = f"""Research Query: {query}

Please work together to answer this query comprehensively:
1. Planner: Create a research plan with specific search queries
2. Researcher: Gather evidence from web and academic sources using your tools
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate quality and signal completion when the response is satisfactory"""

        # Reset team state so internal asyncio queues don't carry over from a prior run
        await self.team.reset()
        result = await self.team.run(task=task_message)

        # Extract conversation history (result.messages is a regular list)
        messages: List[Dict[str, Any]] = []
        for message in result.messages:
            source = getattr(message, "source", "Unknown")
            content = getattr(message, "content", str(message))

            # Handle tool call / tool result messages (content may be a list)
            if isinstance(content, list):
                parts = []
                for item in content:
                    if hasattr(item, "name"):
                        args = getattr(item, "arguments", "")
                        parts.append(f"[Tool call: {item.name}({args})]")
                    elif hasattr(item, "content"):
                        parts.append(str(item.content))
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)

            messages.append({
                "source": source,
                "content": str(content),
                "type": type(message).__name__,
            })

        # Pick the final response: prefer Writer's last *substantive* TextMessage,
        # then Critic's, then the last substantive message overall.
        # We must skip ToolCall/ToolResult messages and tool-call-only stringified
        # content (e.g. "[Tool call: web_search(...)]"), otherwise we may capture
        # the Researcher's tool invocation or the Planner's plan instead of the
        # Writer's synthesized answer.
        def _is_substantive_text(msg: Dict[str, Any]) -> bool:
            mtype = msg.get("type", "")
            if "ToolCall" in mtype or "ToolResult" in mtype:
                return False
            content = msg.get("content", "") or ""
            stripped = content.strip()
            if len(stripped) < 80:
                return False
            # Reject pure tool-call dumps that survived stringification
            if stripped.startswith("[Tool call:") and "[Tool call:" in stripped:
                non_tool = "\n".join(
                    line for line in stripped.splitlines()
                    if not line.strip().startswith("[Tool call:")
                ).strip()
                if len(non_tool) < 80:
                    return False
            return True

        final_response = ""
        for source_pref in ("Writer", "Critic"):
            for msg in reversed(messages):
                if msg.get("source") == source_pref and _is_substantive_text(msg):
                    final_response = msg.get("content", "")
                    break
            if final_response:
                break
        if not final_response:
            for msg in reversed(messages):
                if _is_substantive_text(msg) and msg.get("source") not in (
                    "user",
                    "Planner",
                ):
                    final_response = msg.get("content", "")
                    break
        if not final_response and messages:
            final_response = messages[-1].get("content", "")

        return self._extract_results(query, messages, final_response)

    def _extract_results(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        final_response: str = "",
    ) -> Dict[str, Any]:
        """Structure the conversation into a result dict."""
        research_findings: List[str] = []
        plan = ""
        critique = ""

        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")

            if source == "Planner" and not plan:
                plan = content
            elif source == "Researcher":
                research_findings.append(content)
            elif source == "Critic":
                critique = content

        # Rough source count based on numbered list entries
        num_sources = sum(
            f.count("\n1.") + f.count("\n2.") + f.count("\n3.")
            for f in research_findings
        )

        # Strip TERMINATE marker from final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(
                    {msg.get("source", "") for msg in messages} - {""}
                ),
            },
        }

    # ── Utilities ──────────────────────────────────────────────────────────────

    def get_agent_descriptions(self) -> Dict[str, str]:
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        return """
AutoGen Research Workflow:

1. User Query
   ↓ [Input Safety Check]
2. Planner      → Creates research plan
   ↓
3. Researcher   → Uses web_search() and paper_search() tools
   ↓
4. Writer       → Synthesizes findings with citations
   ↓
5. Critic       → Evaluates; says TERMINATE when satisfied
   ↓ [Output Safety Check]
6. Final Response (sanitized if needed)
"""


def demonstrate_usage():
    """Quick demo of the orchestrator (called by example_autogen.py)."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator = AutoGenOrchestrator(config)
    print(orchestrator.visualize_workflow())

    query = "What are the key principles of explainable AI for novice users?"
    print(f"\nProcessing query: {query}\n" + "=" * 70)

    result = orchestrator.process_query(query)

    print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    meta = result["metadata"]
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {meta.get('num_messages', 0)}")
    print(f"  - Sources gathered:   {meta.get('num_sources', 0)}")
    print(f"  - Agents involved:    {', '.join(meta.get('agents_involved', []))}")
    safety_events = meta.get("safety_events", [])
    print(f"  - Safety events:      {len(safety_events)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    demonstrate_usage()
