"""Shared runtime configuration for the langchain agent project.

Import this module wherever you need the model, console, system prompt,
or database paths.  load_dotenv() is called here once so every script
that imports config automatically picks up the .env file.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rich.console import Console

load_dotenv()

# Local directory that holds all SQLite databases (checkpoints + memory store)
DB_DIR = "agent_db"
CHECKPOINT_DB = os.path.join(DB_DIR, "checkpoints.db")
MEMORY_DB = os.path.join(DB_DIR, "memories.db")

console = Console()

SYSTEM_PROMPT = """You are a research assistant with persistent memory.

Use markdown, colors and emoji in all responses.

Memory guidelines:
- When the user shares preferences or important facts about themselves,
  save them to /memories/user_profile.md
- When you complete a research task, save a summary to /memories/research/<topic>.md
- At the start of each conversation, check /memories/ for relevant context
- Never expose raw file contents — always summarize
"""
model = init_chat_model(
    "azure_openai:gpt-4o-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=1000,
)

