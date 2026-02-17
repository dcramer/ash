"""Observability module for Sentry integration."""

import logging
from typing import TYPE_CHECKING

__all__ = ["init_sentry"]

if TYPE_CHECKING:
    from ash.config import SentryConfig

logger = logging.getLogger(__name__)

try:
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import (
        AsyncioIntegration,
    )
    from sentry_sdk.integrations.logging import (
        LoggingIntegration,
    )

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


def init_sentry(config: "SentryConfig", server_mode: bool = False) -> bool:
    """Initialize Sentry if configured.

    Returns True if Sentry was initialized, False otherwise.
    Server mode enables FastAPI integration.
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
        from sentry_sdk.integrations.fastapi import (
            FastApiIntegration,
        )

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

    logger.info("sentry_initialized", extra={"sentry.environment": config.environment})
    return True
