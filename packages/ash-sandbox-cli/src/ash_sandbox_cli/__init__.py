"""Sandboxed CLI for agent self-service.

This is a minimal, standalone CLI that runs inside the Docker sandbox.
It provides commands the agent can use to manage scheduling and other
tasks without requiring additional tools.

Context is provided via environment variables:
- ASH_SESSION_ID: Current session ID
- ASH_USER_ID: User identifier
- ASH_CHAT_ID: Chat identifier for routing responses
- ASH_PROVIDER: Provider name (e.g., "telegram")
- ASH_USERNAME: Username for @mentions
"""
