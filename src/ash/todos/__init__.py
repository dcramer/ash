"""Todo subsystem public API.

Public API:
- TodoManager: Main entry point
- create_todo_manager: Factory function

Types:
- TodoEntry, TodoEvent, TodoStatus

Spec contract: specs/todos.md.
"""

from ash.todos.manager import TodoManager, create_todo_manager, todo_to_dict
from ash.todos.types import TodoEntry, TodoEvent, TodoStatus, register_todo_graph_schema

__all__ = [
    "TodoManager",
    "TodoEntry",
    "TodoEvent",
    "TodoStatus",
    "register_todo_graph_schema",
    "create_todo_manager",
    "todo_to_dict",
]
