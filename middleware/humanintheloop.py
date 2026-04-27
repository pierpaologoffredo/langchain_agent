"""Demo: HumanInTheLoopMiddleware.

The agent pauses before executing write tools and asks the user to
approve, edit, or reject each action.  Read-only tools run without
interruption.

Supported decisions per action:
  approve — execute the tool as-is
  edit    — override one or more arguments before execution
  reject  — skip the tool call and tell the agent why
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagents import create_deep_agent
from langgraph.types import Command
from rich.markdown import Markdown

from config import checkpointer, console, model
from tools import fetch_wiki_data, save_research_note
from middleware import HumanInTheLoopMiddleware

# InMemorySaver is required for the interrupt/resume flow
agent = create_deep_agent(
    model=model,
    tools=[fetch_wiki_data, save_research_note],
    checkpointer=checkpointer,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_research_note": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "fetch_wiki_data": False,  # read-only — no approval needed
            },
            description_prefix="⚠️ Action requires approval",
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

    # version="v2" is required to receive interrupt events from the middleware
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "main_thread"}},
        version="v2",
    )

    # A single user turn can trigger multiple consecutive interrupts
    while result.interrupts:
        interrupt = result.interrupts[0]
        actions = interrupt.value["action_requests"]

        console.print("\n[bold yellow]⏸  Approval required:[/bold yellow]")
        decisions = []

        for action in actions:
            console.print(f"\n  🔧 Tool: [bold]{action['name']}[/bold]")
            console.print(f"  📋 Args: {action['args']}")
            choice = input("\n  [approve / edit / reject]: ").strip().lower()

            if choice == "approve":
                decisions.append({"type": "approve"})

            elif choice == "edit":
                original = action["args"]
                console.print("\n  [dim]Original args:[/dim]")
                for k, v in original.items():
                    console.print(f"  [dim]  {k}: {v}[/dim]")

                new_args = {}
                for k, v in original.items():
                    console.print(f"\n  Field [bold]{k}[/bold] (press enter to keep original):")
                    console.print(f"  [dim]Current: {v[:80]}{'...' if len(str(v)) > 80 else ''}[/dim]")
                    new_val = input("  New value: ").strip()
                    new_args[k] = new_val if new_val else v

                decisions.append({
                    "type": "edit",
                    "edited_action": {"name": action["name"], "args": new_args},
                })

            elif choice == "reject":
                reason = input("  Reason: ")
                decisions.append({"type": "reject", "message": reason})

        # Resume the agent with the collected decisions
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config={"configurable": {"thread_id": "main_thread"}},
            version="v2",
        )

    # With version="v2", the final messages are nested under result.value
    last_msg = result.value["messages"][-1]
    content = last_msg.content if isinstance(last_msg.content, str) else ""
    console.print("\n[bold green]Agent:[/bold green]")
    console.print(Markdown(content))
    console.print()
