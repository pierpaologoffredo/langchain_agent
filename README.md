# LangChain Agent — Middleware Showcase

This project demonstrates the middleware capabilities of the [`deepagents`](https://pypi.org/project/deepagents/) library, built on top of LangChain and LangGraph. Each script is a self-contained example of a different middleware, showing how to extend and control agent behaviour without modifying core agent logic.

## Middleware examples

| Script | Middleware | What it shows |
|---|---|---|
| `main.py` | `SummarizationMiddleware` | Entry-point agent with weather tool and automatic conversation summarization |
| `middleware/summarization.py` | `SummarizationMiddleware` | Summarizes old messages past a threshold, keeping recent context intact |
| `middleware/humanintheloop.py` | `HumanInTheLoopMiddleware` | Pauses before write operations and asks the user to approve, edit, or reject each action |
| `middleware/calllimit.py` | `ModelCallLimitMiddleware` | Caps the number of model calls per run and raises an error when the limit is exceeded |
| `middleware/todolist.py` | `TodoListMiddleware` | Lets the agent plan and track a structured to-do list across multi-step tasks |

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

## Running an example

```bash
# Main agent (SummarizationMiddleware)
python main.py

# Any middleware example
python middleware/humanintheloop.py
python middleware/summarization.py
python middleware/calllimit.py
python middleware/todolist.py
```

All scripts expose the same interactive CLI. Type `exit` or `quit` to stop.

## Tools available across examples

- **Weather** — current conditions via OpenWeatherMap (`OPENWEATHERMAP_API_KEY` required)
- **Wikipedia** — fetches the top result for a query (no API key needed)
- **Research notes** — saves a note to `test.txt` (used to demonstrate write-tool approval in HITL)
