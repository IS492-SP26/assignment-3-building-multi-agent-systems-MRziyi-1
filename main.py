"""
Main Entry Point

Usage:
  python main.py --mode cli           # Interactive CLI
  python main.py --mode web           # Streamlit web UI
  python main.py --mode evaluate      # Full batch evaluation with LLM-as-a-Judge
  python main.py --mode autogen       # Quick AutoGen demo (default)
"""

import argparse
import asyncio
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_cli():
    """Run interactive CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Launch Streamlit web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(queries_path: str = "data/test_queries.json"):
    """
    Run full batch evaluation with LLM-as-a-Judge.

    Loads test queries from `queries_path`, processes each through the
    multi-agent orchestrator, scores every response with LLMJudge, and
    saves a detailed JSON report + plain-text summary to outputs/.
    """
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("MULTI-AGENT SYSTEM EVALUATION")
    print("=" * 70)

    # Initialize orchestrator
    print("\nInitializing AutoGen orchestrator...")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("Orchestrator ready.")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Initialize evaluator
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    # Run evaluation
    print(f"\nRunning evaluation on: {queries_path}")
    print("This will process each query through all agents and score with LLM-as-a-Judge.\n")

    report = await evaluator.evaluate_system(queries_path)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    summary = report.get("summary", {})
    scores = report.get("scores", {})

    print(f"\nTotal queries:   {summary.get('total_queries', 0)}")
    print(f"Successful:      {summary.get('successful', 0)}")
    print(f"Failed:          {summary.get('failed', 0)}")
    print(f"Success rate:    {summary.get('success_rate', 0.0):.1%}")
    print(f"\nOverall average score: {scores.get('overall_average', 0.0):.3f}")

    print("\nScores by criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        bar = "█" * int(score * 20)
        print(f"  {criterion:<25} {score:.3f}  {bar}")

    best = report.get("best_result")
    worst = report.get("worst_result")
    if best:
        print(f"\nBest  query: [{best['score']:.3f}] {best['query'][:60]}")
    if worst:
        print(f"Worst query: [{worst['score']:.3f}] {worst['query'][:60]}")

    print(f"\nFull results saved to outputs/")


def run_autogen():
    """Quick AutoGen demo (one test query end-to-end)."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant — AI Transparency in HCI"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen"],
        default="autogen",
        help="Mode to run (default: autogen demo)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--queries",
        default="data/test_queries.json",
        help="Path to test queries JSON file (evaluate mode only)",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.queries))
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
