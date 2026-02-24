"""RPC method handlers."""

from ash.rpc.methods.browser import register_browser_methods
from ash.rpc.methods.capability import register_capability_methods
from ash.rpc.methods.config import register_config_methods
from ash.rpc.methods.logs import register_log_methods
from ash.rpc.methods.memory import register_memory_methods
from ash.rpc.methods.schedule import register_schedule_methods
from ash.rpc.methods.todo import register_todo_methods

__all__ = [
    "register_browser_methods",
    "register_capability_methods",
    "register_config_methods",
    "register_log_methods",
    "register_memory_methods",
    "register_schedule_methods",
    "register_todo_methods",
]
