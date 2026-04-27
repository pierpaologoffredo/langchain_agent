"""Demo: ModelCallLimitMiddleware.

Caps the number of LLM calls per agent run and raises when the limit is
exceeded, preventing runaway or looping execution.

The agent is configured with run_limit=2, so the third model call triggers
an exception.  exit_behavior="error" makes the limit immediately visible
during testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_agent
from rich.markdown import Markdown

from config import checkpointer, console, model
from tools import ALL_TOOLS
from middleware import ModelCallLimitMiddleware

agent = create_agent(
    model=model,
    tools=ALL_TOOLS,
    checkpointer=checkpointer,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=2,           # error on the 3rd model call within a single run
            exit_behavior="error",
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

    if not user_input or user_input.lower() in ("exit", "quit"):
        console.print("\n[bold red]Exiting...[/bold red]")
        break

    try:
        # version="v2" is required to receive interrupt/error events
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "main_thread"}},
            version="v2",
        )

        # With version="v2", messages are nested under result.value
        last_msg = result.value["messages"][-1]
        content = last_msg.content if isinstance(last_msg.content, str) else ""
        console.print("\n[bold green]Agent:[/bold green]")
        console.print(Markdown(content))
        console.print()

    except Exception as e:
        console.print(f"[bold red]Call limit reached: {e}[/bold red]")
