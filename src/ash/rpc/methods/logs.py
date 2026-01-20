"""Log query RPC method handlers."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer

logger = logging.getLogger(__name__)


def register_log_methods(server: "RPCServer", logs_path: Path) -> None:
    """Register log-related RPC methods.

    Args:
        server: RPC server to register methods on.
        logs_path: Path to the logs directory.
    """

    async def logs_query(params: dict[str, Any]) -> list[dict[str, Any]]:
        """Query log files and return matching entries.

        Params:
            since: ISO timestamp or relative time (1h, 30m, 1d) - optional
            until: ISO timestamp - optional
            search: Text to search for in messages - optional
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR) - optional
            component: Component name filter - optional
            limit: Maximum entries (default 50)
        """
        from ash.cli.commands.logs import parse_level, parse_time, query_logs

        # Parse time parameters
        since = params.get("since")
        since_dt = parse_time(since) if since else None

        until = params.get("until")
        until_dt = parse_time(until) if until else None

        # Parse level
        level = params.get("level")
        level_value = parse_level(level) if level else None

        # Other filters
        search_pattern = params.get("search")
        component = params.get("component")
        limit = params.get("limit", 50)

        # Query logs
        return query_logs(
            logs_path,
            since=since_dt,
            until=until_dt,
            search_pattern=search_pattern,
            level_value=level_value,
            component=component,
            limit=limit,
        )

    server.register("logs.query", logs_query)
    logger.debug("Registered log RPC methods")
