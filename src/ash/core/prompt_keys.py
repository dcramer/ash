"""Structured prompt extension keys for integration hooks.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

CORE_PRINCIPLES_RULES_KEY = "core_principles_rules"
TOOL_ROUTING_RULES_KEY = "tool_routing_rules"

RESERVED_PROMPT_EXTENSION_KEYS = {
    CORE_PRINCIPLES_RULES_KEY,
    TOOL_ROUTING_RULES_KEY,
}
