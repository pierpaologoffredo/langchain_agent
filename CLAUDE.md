# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Agent

```bash
python main.py
```

Uses `uv` as the package manager (Python 3.13+):

```bash
uv sync          # install dependencies
uv run main.py   # run via uv
```

## Required Environment Variables

Create a `.env` file with:

- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI deployment URL
- `OPENAI_API_VERSION` — e.g. `2024-12-01-preview`
- `OPENAI_API_KEY` — OpenAI/Azure API key
- `OPENWEATHERMAP_API_KEY` — For weather tool
- `LANGSMITH_API_KEY` — LangChain tracing
- `LANGSMITH_TRACING` — Set to `true` to enable tracing

## Architecture

The entire application lives in `main.py`. Data flow:

1. User types input in the CLI loop
2. Input is passed to `deep_agent.invoke()` (from `deepagents.create_deep_agent`)
3. The agent uses `azure_openai:gpt-4o-mini` via `init_chat_model()`
4. Available tool: `OpenWeatherMapAPIWrapper.run` for weather queries
5. Conversation state persists in-memory via `langgraph`'s `InMemorySaver` (single `thread_id="main_thread"`)
6. `SummarizationMiddleware` summarizes history after every 2 messages, retaining the last 20
7. Response is rendered as Markdown using `rich`

## Key Libraries

- **`deepagents`** — wraps LangGraph to create the agent loop (`create_deep_agent`)
- **`langchain`** / **`langchain-community`** — LLM orchestration and tool integrations
- **`langgraph`** — provides `InMemorySaver` for conversation checkpointing
- **`rich`** — terminal Markdown rendering
