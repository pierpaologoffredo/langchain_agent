# Demonstrates HumanInTheLoopMiddleware: the agent pauses before executing write tools
# and asks the user to approve, edit, or reject each action before proceeding.

# --- Imports ---

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from deepagents import create_deep_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from rich.console import Console
from rich.markdown import Markdown

# --- Setup ---

load_dotenv()

console = Console()

# Wikipedia retriever used by the fetch tool
retriever = WikipediaRetriever(
    wiki_client="",
    top_k_results=1,
    doc_content_chars_max=10000,
)

model = init_chat_model("azure_openai:gpt-4o-mini", temperature=0.5)

# InMemorySaver is required for interrupt/resume support
checkpointer = InMemorySaver()

# --- Tools ---

@tool
def fetch_wiki_data(query: str) -> str:
    """Fetch content of Wikipedia page from top hit of a query."""
    res = retriever.invoke(query)
    if res:
        return res[0].page_content
    return "No data found"

@tool
def save_research_note(note: str, topic: str) -> str:
    """Save an important research note about a topic to the knowledge base."""
    with open("test.txt", "w", encoding="utf-8") as txt:
        txt.write(f"Note saved — Topic: {topic} | Content: {note}")
    return f"Note saved — Topic: {topic} | Content: {note}"

# --- Agent ---

# Only write tools require approval; read-only tools run without interruption.
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
                "fetch_wiki_data": False,  # read-only, no approval needed
            },
            description_prefix="⚠️ Action requires approval",
        ),
    ],
)

# --- Main loop ---

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

    # version="v2" is required to receive interrupt events
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

                # Allow the user to override each argument individually
                new_args = {}
                for k, v in original.items():
                    console.print(f"\n  Field [bold]{k}[/bold] (press enter to keep original):")
                    console.print(f"  [dim]Current: {v[:80]}{'...' if len(str(v)) > 80 else ''}[/dim]")
                    new_val = input("  New value: ").strip()
                    new_args[k] = new_val if new_val else v

                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": action["name"],
                        "args": new_args,
                    },
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
