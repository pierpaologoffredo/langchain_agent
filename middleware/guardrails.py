"""Custom guardrail middleware: deterministic input filter + LLM-based output check.

Both classes implement AgentMiddleware hooks and can be composed freely
with the built-in middleware (PIIMiddleware, SummarizationMiddleware, etc.).
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
class InputGuardrailMiddleware(AgentMiddleware):
    """Deterministic input guardrail: blocks banned keywords and oversized messages.

    Runs *before* the model.  If the latest user message contains a banned
    keyword or exceeds `max_length` characters, the middleware short-circuits
    the pipeline and returns an error reply without ever calling the LLM.

    Args:
        banned_keywords: Words or phrases that must not appear in user input.
                         Matching is case-insensitive.
        max_length:      Maximum allowed character count for a single message.
                         Defaults to 1 000.
    """

    def __init__(self, banned_keywords: list[str], max_length: int = 1_000):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
        self.max_length = max_length

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last = state["messages"][-1]
        if last.type != "human":
            return None

        content = last.content

        if len(content) > self.max_length:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"⚠️ Input too long ({len(content)} characters). "
                        f"Maximum allowed: {self.max_length}."
                    ),
                }],
                "jump_to": "end",
            }

        content_lower = content.lower()
        for keyword in self.banned_keywords:
            if keyword in content_lower:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "⚠️ I cannot process this request as it contains "
                            "disallowed content. Please rephrase your message."
                        ),
                    }],
                    "jump_to": "end",
                }

        return None


class OutputSafetyMiddleware(AgentMiddleware):
    """LLM-based output guardrail: validates the model's response before delivery.

    Runs *after* the model.  Passes the reply to a safety judge (a smaller
    LLM) that classifies it as SAFE or UNSAFE.  Unsafe responses are replaced
    with a generic refusal so the raw output never reaches the user.

    The judge is given the list of registered tools so it can correctly
    evaluate whether the response stays within the agent's intended scope.

    Args:
        tools:           Tool list registered with the agent (defines scope).
        safety_model_id: LangChain model ID for the safety judge.
                         Defaults to "azure_openai:gpt-4o-mini".
    """

    def __init__(self, tools: list, safety_model_id: str = "azure_openai:gpt-4o-mini"):
        super().__init__()
        self.safety_model = init_chat_model(safety_model_id, temperature=0.3)
        # Build the tools description once at init to avoid repeating it per call
        self.tools_context = "\n".join(
            f"- {t.name if hasattr(t, 'name') else t.__name__}: "
            f"{t.description if hasattr(t, 'description') else 'no description'}"
            for t in tools
        )

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None

        safety_prompt = f"""You are a safety evaluator for an AI assistant.

The assistant has access to the following tools, which define its operational scope:
{self.tools_context}

Given this context, evaluate whether the response below is safe and appropriate.
Flag as UNSAFE only if the response:
- Assists with tasks clearly outside the tools' intended scope
- Exposes internal system data (e.g. raw API keys, internal URLs, stack traces)
- Contains harmful, misleading, or manipulative content
- Reveals private user data not sourced from the tools themselves

Do NOT flag as UNSAFE:
- Factual public information retrieved by the tools (e.g. Wikipedia content, weather data)
- Biographical data about public figures (birth dates, career history, etc.)
- Summaries or elaborations of tool results

Reply ONLY with 'SAFE' or 'UNSAFE: <brief reason>'.

Response to evaluate:
{last.content}"""

        judgment = self.safety_model.invoke(
            [{"role": "user", "content": safety_prompt}],
            config={"callbacks": []},
        )

        if judgment.content.startswith("UNSAFE"):
            reason = judgment.content.replace("UNSAFE: ", "")
            last.content = (
                f"⚠️ I cannot provide this response ({reason}). "
                "Please try rephrasing your request."
            )

        return None
