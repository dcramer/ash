"""RPC method handlers."""

from ash.rpc.methods.config import register_config_methods
from ash.rpc.methods.memory import register_memory_methods

__all__ = ["register_config_methods", "register_memory_methods"]
