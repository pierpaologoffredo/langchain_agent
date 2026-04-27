"""Demo: TodoListMiddleware.

The agent automatically writes a structured to-do list before tackling
multi-step requests.  Tasks are ordered logically: research first, then
elaboration, finally write operations that may require approval.

After each turn the completed task list is printed below the response.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_agent
from rich.markdown import Markdown

from config import checkpointer, console, model, SYSTEM_PROMPT
from tools import ALL_TOOLS
from middleware import TodoListMiddleware

# This prompt is *appended* to the middleware's built-in system prompt —
# it does not replace it.  Use it to steer when and how todos are created.
CUSTOM_TODO_PROMPT = """
Use write_todos when:
- The task needs more than 3 tools
- You need to find information and then elaborate on it
- The user gives you a to-do list

Structure todos in a logical order: research first, then processing,
finally write operations that may require approval.
"""

agent = create_agent(
    model=model,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        TodoListMiddleware(system_prompt=CUSTOM_TODO_PROMPT),
    ],
)

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

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "main_thread"}},
    )

    # Display the status of each task from the todo list
    todos = result.get("todos", [])
    if todos:
        console.print("\n[bold cyan]📋 Tasks:[/bold cyan]")
        icons = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}
        for todo in todos:
            icon = icons.get(todo["status"], "•")
            console.print(f"  {icon} {todo['content']}")

    last_msg = result["messages"][-1]
    content = last_msg.content if isinstance(last_msg.content, str) else last_msg.content[0].get("text", "")
    console.print("\n[bold green]Agent:[/bold green]")
    console.print(Markdown(content))
    console.print()
