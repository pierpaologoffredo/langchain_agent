# Demonstrates ModelCallLimitMiddleware: caps the number of model calls per agent run
# and raises an error when the limit is exceeded, preventing runaway execution.

# --- Imports ---

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown

# --- Setup ---

load_dotenv()

console = Console()

model = init_chat_model("azure_openai:gpt-4o-mini", temperature=0.5)

checkpointer = InMemorySaver()

# --- Tools ---

weather = OpenWeatherMapAPIWrapper()

retriever = WikipediaRetriever(
    wiki_client="",
    top_k_results=1,
    doc_content_chars_max=10000,
)

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

# run_limit=2 means the agent errors out on the 3rd model call.
# exit_behavior="error" raises an exception, making the limit immediately visible.
agent = create_agent(
    model=model,
    tools=[fetch_wiki_data, save_research_note, weather.run],
    checkpointer=checkpointer,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=2,
            exit_behavior="error",
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

    try:
        # version="v2" is required to receive interrupt events
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