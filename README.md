# LangChain Agent — Middleware Showcase

A modular LangChain/LangGraph agent that demonstrates the middleware capabilities of the [`deepagents`](https://pypi.org/project/deepagents/) library. Tools, middleware, and configuration live in separate modules so each component can be reused or swapped without touching the rest of the codebase.

## Project structure

```
langchain_agent/
├── main.py                    # Entry point — CLI loop with the full safety stack
├── config.py                  # Shared config: model, console, DB paths, system prompt
├── tools/
│   ├── __init__.py            # Exports ALL_TOOLS and individual tools
│   ├── weather.py             # get_weather — current conditions via OpenWeatherMap
│   ├── wikipedia.py           # fetch_wiki_data — top Wikipedia result for a query
│   └── notes.py               # save_research_note — persists a finding to test.txt
├── store/
│   ├── __init__.py            # Exports SQLiteStore
│   └── sqlite.py              # SQLiteStore — BaseStore backed by a local SQLite file
└── middleware/
    ├── __init__.py            # Re-exports all middleware from one location
    ├── guardrails.py          # Custom: InputGuardrailMiddleware + OutputSafetyMiddleware
    ├── pii.py                 # Demo: PII filtering + guardrails stack
    ├── summarization.py       # Demo: automatic conversation summarization
    ├── calllimit.py           # Demo: model call cap per run
    ├── humanintheloop.py      # Demo: human approval before write operations
    └── todolist.py            # Demo: structured to-do list planning
```

## Setup

**Requirements:** Python 3.13+, [`uv`](https://github.com/astral-sh/uv)

```bash
uv sync
```

Create a `.env` file in the project root:

```env
AZURE_OPENAI_ENDPOINT=...
OPENAI_API_VERSION=2024-12-01-preview
OPENAI_API_KEY=...
OPENWEATHERMAP_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
```

## Running the agent

```bash
# Main agent — full PII + guardrails safety stack
python main.py

# Individual middleware demos
python middleware/pii.py
python middleware/summarization.py
python middleware/calllimit.py
python middleware/humanintheloop.py
python middleware/todolist.py
```

All scripts expose the same interactive CLI. Type `exit`, `quit`, or `bye` to stop.

## Middleware overview

| Script | Middleware | What it shows |
|---|---|---|
| `main.py` / `middleware/pii.py` | `PIIMiddleware` + `InputGuardrailMiddleware` + `OutputSafetyMiddleware` | Full safety stack: PII redaction/masking/blocking, keyword filter, LLM-based output check |
| `middleware/summarization.py` | `SummarizationMiddleware` | Compresses old messages past a threshold, keeping recent context intact |
| `middleware/humanintheloop.py` | `HumanInTheLoopMiddleware` | Pauses before write operations and asks the user to approve, edit, or reject each action |
| `middleware/calllimit.py` | `ModelCallLimitMiddleware` | Caps the number of LLM calls per run and raises when the limit is exceeded |
| `middleware/todolist.py` | `TodoListMiddleware` | Lets the agent plan and track a structured to-do list across multi-step tasks |

## Custom middleware

Two custom guardrail classes are defined in `middleware/guardrails.py` and available via `from middleware import ...`:

- **`InputGuardrailMiddleware`** — deterministic filter that runs before the model. Rejects messages containing banned keywords or exceeding a character limit.
- **`OutputSafetyMiddleware`** — LLM-based judge that runs after the model. Classifies the response as `SAFE` or `UNSAFE` and replaces unsafe replies with a generic refusal.

## Extending the project

**New tool** — create `tools/<name>.py` with a `@tool`-decorated function, then import it in `tools/__init__.py` and add it to `ALL_TOOLS`.

**New middleware** — add a custom class to `middleware/guardrails.py` (or a new file), re-export it from `middleware/__init__.py`, and include it in the `middleware=[...]` list in `main.py` or the relevant demo.

**New store backend** — add a new file under `store/` that subclasses `BaseStore`, export it from `store/__init__.py`, and pass it as `store=` to `create_deep_agent`.  The DB paths live in `config.py` (`DB_DIR`, `CHECKPOINT_DB`, `MEMORY_DB`).
