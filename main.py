"""Entry point for the langchain agent CLI.

Builds a DeepAgent with the full safety stack (PII filtering, output safety
check, and input guardrails) and starts an interactive CLI loop.

Type 'exit', 'quit', or 'bye' — or press Ctrl+C — to stop the agent.
"""

import os

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.markdown import Markdown

from config import CHECKPOINT_DB, DB_DIR, MEMORY_DB, console, model, SYSTEM_PROMPT
from store import SQLiteStore
from tools import ALL_TOOLS
from middleware import (
    InputGuardrailMiddleware,
    OutputSafetyMiddleware,
    PIIMiddleware,
)

os.makedirs(DB_DIR, exist_ok=True)

store = SQLiteStore(db_path=MEMORY_DB)

def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(
                runtime,
                namespace=lambda ctx: ("memories", "pierpaolo"),
            )
        }
    )

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:


    deep_agent = create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        middleware=[
            # Strip email addresses from both user input and model output
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
            # Block messages that contain an OpenAI-style API key
            PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32,}", strategy="block", apply_to_input=True),
            # Mask credit card numbers before they reach the model
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            # Redact URLs from both sides of the conversation
            PIIMiddleware("url", strategy="redact", apply_to_input=True, apply_to_output=True),
            # LLM-based safety check on the model's final answer
            OutputSafetyMiddleware(safety_model_id="azure_openai:gpt-4o-mini", tools=ALL_TOOLS),
            # Keyword + length filter — runs before the model, short-circuits on violations
            InputGuardrailMiddleware(
                banned_keywords=[
                    "hack", "exploit", "jailbreak",
                    "ignore previous", "ignore instructions",
                    "disregard", "bypass",
                ],
                max_length=1_000,
            ),
        ],
    )

    console.print("[bold cyan]Assistant ready. Type [bold]exit[/bold] to quit.[/bold cyan]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold red]Exiting...[/bold red]")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "bye"):
            console.print("\n[bold red]Exiting...[/bold red]")
            break

        result = deep_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "main_thread"}},
        )

        last_msg = result["messages"][-1]
        content = last_msg.content if isinstance(last_msg.content, str) else last_msg.content[0].get("text", "")
        console.print("\n[bold green]Agent:[/bold green]")
        console.print(Markdown(content))
        console.print()
