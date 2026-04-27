"""Shared runtime configuration for the langchain agent project.

Import this module wherever you need the model, checkpointer, console,
or the default system prompt.  load_dotenv() is called here once so
every script that imports config automatically picks up the .env file.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

load_dotenv()

console = Console()

SYSTEM_PROMPT = "Print the answer using markdown, colors and emoji."

model = init_chat_model(
    "azure_openai:gpt-4o-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=1000,
)

# In-memory conversation checkpointer — swappable with SqliteSaver for persistence
checkpointer = InMemorySaver()
