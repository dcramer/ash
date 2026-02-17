"""RPC method handlers."""

from ash.rpc.methods.config import register_config_methods
from ash.rpc.methods.logs import register_log_methods
from ash.rpc.methods.memory import register_memory_methods
from ash.rpc.methods.schedule import register_schedule_methods

__all__ = [
    "register_config_methods",
    "register_log_methods",
    "register_memory_methods",
    "register_schedule_methods",
]
