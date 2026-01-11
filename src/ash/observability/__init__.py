"""Observability module for Sentry integration."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.config import SentryConfig

logger = logging.getLogger(__name__)

# Check availability
try:
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


def init_sentry(config: "SentryConfig", server_mode: bool = False) -> bool:
    """Initialize Sentry if configured.

    Args:
        config: Sentry configuration.
        server_mode: Whether running in server mode (enables FastAPI integration).

    Returns:
        True if Sentry was initialized, False otherwise.
    """
    if not SENTRY_AVAILABLE:
        logger.debug("Sentry SDK not installed, skipping initialization")
        return False

    if not config.dsn:
        logger.debug("Sentry DSN not configured, skipping initialization")
        return False

    integrations = [
        AsyncioIntegration(),
        LoggingIntegration(
            level=logging.INFO,  # Capture INFO+ as breadcrumbs
            event_level=logging.ERROR,  # Create events for ERROR+
        ),
    ]

    if server_mode:
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        integrations.append(FastApiIntegration())

    sentry_sdk.init(
        dsn=config.dsn.get_secret_value(),
        environment=config.environment,
        release=config.release,
        traces_sample_rate=config.traces_sample_rate,
        profiles_sample_rate=config.profiles_sample_rate,
        send_default_pii=config.send_default_pii,
        debug=config.debug,
        integrations=integrations,
    )

    logger.info(f"Sentry initialized (environment={config.environment})")
    return True
