# Demonstrates SummarizationMiddleware: automatically summarizes older messages once a
# configured threshold is reached, keeping recent ones intact and preserving AI/Tool message pairs.

# --- Imports ---

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain.agents.middleware import SummarizationMiddleware
from rich.console import Console
from rich.markdown import Markdown

# --- Setup ---

load_dotenv()

console = Console()

SYSTEM_PROMPT = "Print the answer using markdown, colors and emoji."

model = init_chat_model(
    "azure_openai:gpt-4o-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=1000,
)

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

# --- Agent ---

# Custom prompt used by SummarizationMiddleware when compressing old messages
summary_prompt = """
Summarize the main thrust of this conversation. What have the human and assistant discussed so far? Focus on key facts and requests.
<messages>
Messages to summarize:
{messages}
</messages>
"""

deep_agent = create_deep_agent(
    model=model,
    tools=[weather.run, fetch_wiki_data],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model=model,
            summary_prompt=summary_prompt,
            trigger=("messages", 3),   # low threshold, suitable for testing
            keep=("messages", 1),      # retain only the latest message after summarization
            trim_tokens_to_summarize=None,
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
