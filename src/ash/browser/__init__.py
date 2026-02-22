"""Browser subsystem.

Public API:
- BrowserManager: Browser action/session facade
- create_browser_manager: Factory for manager wiring

Types:
- BrowserSession, BrowserProfile, BrowserActionResult
"""

from ash.browser.manager import (
    BrowserManager,
    create_browser_manager,
    format_browser_result,
)
from ash.browser.types import BrowserActionResult, BrowserProfile, BrowserSession

__all__ = [
    "BrowserManager",
    "create_browser_manager",
    "format_browser_result",
    "BrowserActionResult",
    "BrowserProfile",
    "BrowserSession",
]
