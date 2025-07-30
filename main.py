#!/usr/bin/env python3
"""
Updated entry point script for the DeerFlow project.
Uses the new optimized configuration system with Pydantic models.
"""

import argparse
import asyncio
import logging

from InquirerPy import inquirer

from src.config import load_configuration
from src.constants.questions import BUILT_IN_QUESTIONS
from src.workflow import run_agent_workflow_async

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_configuration(config_path: str = "conf.yaml"):
    """Initialize the new configuration system."""
    try:
        # Load configuration from specified path
        settings = load_configuration(config_path)
        logger.info(f"Configuration loaded successfully from {config_path}")
        logger.info(f"Configuration summary: {settings.report_style}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        # Fallback to default configuration
        return load_configuration("conf.yaml")


def ask(
    question,
    debug=False,
    max_plan_iterations=None,
    max_step_num=None,
    enable_background_investigation=True,
    enable_collaboration=True,
    config_path="conf.yaml",
    locale="ru-RU",
):
    """Run the agent workflow with the given question using new configuration.

    Args:
        question: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations (overrides config)
        max_step_num: Maximum number of steps in a plan (overrides config)
        enable_background_investigation: If True, performs web search before planning
        enable_collaboration: If True, enables collaboration features
        config_path: Path to configuration file
    """
    # Initialize configuration
    settings = initialize_configuration(config_path)

    # Override configuration if provided
    if max_plan_iterations is not None:
        settings.agents.max_plan_iterations = max_plan_iterations
    if max_step_num is not None:
        settings.agents.max_step_num = max_step_num

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Configuration: {settings.dict()}")

    asyncio.run(
        run_agent_workflow_async(
            user_input=question,
            debug=debug,
            max_step_num=max_step_num,
            settings=settings,
            enable_background_research=enable_background_investigation,
            enable_collaboration=enable_collaboration,
            locale=locale,
        )
    )


def main(
    debug=False,
    max_plan_iterations=None,
    max_step_num=None,
    enable_background_investigation=True,
    enable_collaboration=True,
    config_path="conf.yaml",
):
    """Interactive mode with built-in questions using new configuration.

    Args:
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning
        enable_collaboration: If True, enables collaboration features
        config_path: Path to configuration file
    """
    # Initialize configuration
    settings = initialize_configuration(config_path)

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Configuration: {settings.dict()}")

    # Use Russian as default language
    language = "Русский"
    locale = "ru-RU"

    # Use Russian questions
    questions = BUILT_IN_QUESTIONS
    ask_own_option = "[Задать свой вопрос]"

    # Select a question
    initial_question = inquirer.select(
        message="Что вы хотите узнать?",
        choices=[ask_own_option] + questions,
    ).execute()

    if initial_question == ask_own_option:
        initial_question = inquirer.text(
            message="Что вы хотите узнать?",
        ).execute()

    # Run with configuration
    ask(
        question=initial_question,
        debug=debug,
        max_plan_iterations=max_plan_iterations,
        max_step_num=max_step_num,
        enable_background_investigation=enable_background_investigation,
        enable_collaboration=enable_collaboration,
        config_path=config_path,
        locale=locale,
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="DeerFlow - Advanced AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the latest developments in quantum computing?"
  %(prog)s --interactive
  %(prog)s --config custom.yaml "Research renewable energy trends"
  %(prog)s --debug --max-plan-iterations 5 "Analyze AI safety concerns"
        """,
    )

    parser.add_argument("query", nargs="*", help="The research query to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with built-in questions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf.yaml",
        help="Configuration file path (default: conf.yaml)",
    )
    parser.add_argument(
        "--max-plan-iterations",
        type=int,
        help="Maximum number of plan iterations (overrides config)",
    )
    parser.add_argument(
        "--max-step-num",
        type=int,
        help="Maximum number of steps in a plan (overrides config)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-background-investigation",
        action="store_false",
        dest="enable_background_investigation",
        help="Disable background investigation before planning",
    )

    args = parser.parse_args()

    if args.interactive:
        # Pass command line arguments to main function
        main(
            debug=args.debug,
            max_plan_iterations=args.max_plan_iterations,
            max_step_num=args.max_step_num,
            enable_background_investigation=args.enable_background_investigation,
            config_path=args.config,
        )
    else:
        # Parse user input from command line arguments or user input
        if args.query:
            user_query = " ".join(args.query)
        else:
            user_query = input("Enter your query: ")

        if user_query.strip():
            ask(
                question=user_query,
                debug=args.debug,
                max_plan_iterations=args.max_plan_iterations,
                max_step_num=args.max_step_num,
                enable_background_investigation=args.enable_background_investigation,
                config_path=args.config,
            )
        else:
            print("No query provided. Use --interactive for guided mode.")
