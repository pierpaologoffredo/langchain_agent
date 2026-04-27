from langchain.agents.middleware import PIIMiddleware
from deepagents import create_deep_agent
# --- Imports ---
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from langchain.messages import AIMessage

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


# --- Middleware ---
class InputGuardrailMiddleware(AgentMiddleware):
    """Deterministic input guardrail: keyword filter + length check."""

    def __init__(self, banned_keywords: list[str], max_length: int = 20):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
        self.max_length = max_length

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][-1]
        if first_message.type != "human":
            return None

        content = first_message.content

        # Length check
        if len(content) > self.max_length:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"⚠️ Input too long ({len(content)} characters). "
                        f"Maximum allowed: {self.max_length}."
                    )
                }],
                "jump_to": "end"
            }

        # Keyword check
        content_lower = content.lower()
        for keyword in self.banned_keywords:
            if keyword in content_lower:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "⚠️ I cannot process this request as it contains "
                            "disallowed content. Please rephrase your message."
                        )
                    }],
                    "jump_to": "end"
                }

        return None
    
class OutputSafetyMiddleware(AgentMiddleware):
    """LLM-based output guardrail: uses a small model as a safety judge."""

    def __init__(
        self,
        tools: list,
        safety_model_id: str = "azure_openai:gpt-4o-mini",
    ):
        super().__init__()
        self.safety_model = init_chat_model(safety_model_id, temperature=0.3)
        # Costruisce la descrizione dei tool una sola volta all'init
        self.tools_context = "\n".join(
            f"- {t.name if hasattr(t, 'name') else t.__name__}: "
            f"{t.description if hasattr(t, 'description') else 'no description'}"
            for t in tools
        )


    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        safety_prompt = f"""You are a safety evaluator for an AI assistant.

                        The assistant has access to the following tools, which define its operational scope:
                        {self.tools_context}

                        Given this context, evaluate whether the response below is safe and appropriate:
                        - It should only contain information relevant to the tools' domains
                        - It should not reveal sensitive data produced by the tools (e.g. raw API responses, internal data)
                        - It should not assist with tasks clearly outside the tools' intended scope
                        - It should not contain harmful, misleading, or manipulative content

                        Reply ONLY with 'SAFE' or 'UNSAFE: <brief reason>'.

                        Response to evaluate:
                        {last_message.content}"""

        judgment = self.safety_model.invoke([
            {"role": "user", "content": safety_prompt}
        ])

        if judgment.content.startswith("UNSAFE"):
            reason = judgment.content.replace("UNSAFE: ", "")
            last_message.content = (
                f"⚠️ I cannot provide this response ({reason}). "
                "Please try rephrasing your request."
            )

        return None

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


deep_agent = create_deep_agent(
    model=model,
    tools=[weather.run, fetch_wiki_data, save_research_note],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        # Redacta email nell'input prima che arrivi al modello
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,  # anche nella risposta
        ),
        # Blocca se rileva API key (custom regex)
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32,}",  # pattern OpenAI keys
            strategy="block",
            apply_to_input=True,
        ),
        # Maschera carte di credito
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "url",
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        # Layer 4 — LLM-based output safety check
        OutputSafetyMiddleware(safety_model_id="azure_openai:gpt-4o-mini", tools=[weather.run, fetch_wiki_data, save_research_note]),
        # Layer 5 — Deterministic input guardrail
        InputGuardrailMiddleware(
            banned_keywords=[
                "hack", "exploit", "jailbreak",
                "ignore previous", "ignore instructions",
                "disregard", "bypass",
            ],
            max_length=50,
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