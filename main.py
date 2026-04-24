from langchain.agents import create_agent
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import os
from deepagents import create_deep_agent
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from rich.console import Console
from rich.markdown import Markdown
from langchain.agents.middleware import SummarizationMiddleware

console = Console()

load_dotenv()

checkpointer = InMemorySaver()

weather = OpenWeatherMapAPIWrapper()

SYSTEM_PROMPT = "Print the answer using markdown, colors and emoji."

model = init_chat_model(
    "azure_openai:gpt-4o-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=1000,
)


deep_agent = create_deep_agent(
    model=model,
    tools = [weather.run],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 2),
            keep=("messages", 20),
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