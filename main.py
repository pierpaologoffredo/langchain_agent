"""Entry point for the langchain agent CLI.

Builds a DeepAgent with the full safety stack (PII filtering, output safety
check, and input guardrails) and starts an interactive CLI loop.

Type 'exit', 'quit', or 'bye' — or press Ctrl+C — to stop the agent.
"""

import os

import openai
from deepagents import create_deep_agent
from langchain_core.messages import AIMessageChunk
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHECKPOINT_DB, DB_DIR, MEMORY_DB, model, SYSTEM_PROMPT
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

    print("Assistant ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "bye"):
            print("\nExiting...")
            break

        print("\nAgent: ", end="", flush=True)
        try:
            for chunk, _ in deep_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": "main_thread"}},
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
                    print(chunk.content, end="", flush=True)
        except openai.BadRequestError as e:
            if e.code == "content_filter":
                print("Your message was flagged by the content filter. Please rephrase and try again.")
            else:
                print(f"Request error: {e}")
        except openai.RateLimitError:
            print("The API rate limit was hit. Please wait a moment and try again.")
        print("\n")
