"""Demo: SummarizationMiddleware.

Automatically compresses older messages once the conversation exceeds a
configured threshold, keeping recent ones intact and preserving AI/Tool
message pairs so the agent never loses the most recent context.

Run directly to start the agent with summarization enabled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagents import create_deep_agent
from rich.markdown import Markdown

from config import checkpointer, console, model, SYSTEM_PROMPT
from tools import fetch_wiki_data, get_weather
from middleware import SummarizationMiddleware

# Prompt sent to the summarization model when compressing old messages.
# The {messages} placeholder is filled in by the middleware.
SUMMARY_PROMPT = """
Summarize the main thrust of this conversation. What have the human and assistant discussed so far? Focus on key facts and requests.
<messages>
Messages to summarize:
{messages}
</messages>
"""

deep_agent = create_deep_agent(
    model=model,
    tools=[get_weather, fetch_wiki_data],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model=model,
            summary_prompt=SUMMARY_PROMPT,
            trigger=("messages", 3),   # low threshold — convenient for manual testing
            keep=("messages", 1),      # retain only the latest message after summarization
            trim_tokens_to_summarize=None,
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
