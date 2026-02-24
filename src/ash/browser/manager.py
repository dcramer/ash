"""Browser subsystem manager and factory."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.browser.providers.base import BrowserProvider, ProviderGotoResult
from ash.browser.providers.kernel import KernelBrowserProvider
from ash.browser.providers.sandbox import SandboxBrowserProvider
from ash.browser.store import BrowserStore
from ash.browser.types import BrowserActionResult, BrowserProfile, BrowserSession
from ash.config import AshConfig
from ash.config.paths import get_browser_path

if TYPE_CHECKING:
    from ash.sandbox.executor import SandboxExecutor

logger = logging.getLogger("browser")


class BrowserManager:
    """Coordinates browser sessions/actions across providers."""

    def __init__(
        self,
        *,
        config: AshConfig,
        store: BrowserStore,
        providers: dict[str, BrowserProvider],
    ) -> None:
        self._config = config
        self._store = store
        self._providers = providers

    @property
    def store(self) -> BrowserStore:
        return self._store

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Configured browser providers available at runtime."""
        return tuple(sorted(self._providers.keys()))

    def _is_sandbox_runtime(self) -> bool:
        """Return True when running in sandbox/container runtime context."""
        env_value = (
            os.environ.get("ASH_BROWSER_SANDBOX_RUNTIME")
            or os.environ.get("ASH_IN_SANDBOX")
            or ""
        ).strip()
        if env_value.lower() in {"1", "true", "yes", "on"}:
            return True
        return Path("/.dockerenv").exists()

    async def warmup_default_provider(self) -> None:
        """Best-effort warmup for configured default provider."""
        if not self._config.browser.enabled:
            return
        provider_key = self._config.browser.provider.strip().lower()
        provider = self._providers.get(provider_key)
        if provider is None:
            return
        warmup = getattr(provider, "warmup", None)
        if not callable(warmup):
            return
        logger.info(
            "browser_runtime_starting",
            extra={"browser.provider": provider_key},
        )
        try:
            await warmup()
        except Exception as e:
            logger.warning(
                "browser_runtime_start_failed",
                extra={
                    "browser.provider": provider_key,
                    "error.message": str(e),
                },
            )
            return
        logger.info(
            "browser_runtime_ready",
            extra={"browser.provider": provider_key},
        )

    async def shutdown(self) -> None:
        """Best-effort shutdown for provider runtimes during service teardown."""
        for provider_key, provider in self._providers.items():
            shutdown = getattr(provider, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                await shutdown()
            except Exception as e:
                logger.warning(
                    "browser_runtime_shutdown_failed",
                    extra={
                        "browser.provider": provider_key,
                        "error.message": str(e),
                    },
                )

    def _action_timeout_seconds(self, action: str) -> float:
        base = self._clamp_timeout_seconds(float(self._config.browser.timeout_seconds))
        if action == "session.start":
            return min(60.0, max(15.0, base + 10.0))
        if action.startswith("page."):
            return min(120.0, max(10.0, base + 5.0))
        return min(60.0, max(10.0, base + 5.0))

    def _clamp_timeout_seconds(self, value: float) -> float:
        return max(1.0, min(120.0, value))

    def _normalize_browser_error_message(self, message: str) -> str:
        runtime_unavailable_markers = (
            "sandbox_runtime_required",
            "sandbox_executor_required",
            "sandbox_browser_launch_failed",
            "sandbox_browser_runtime_unavailable",
            "cdp_not_ready",
        )
        if any(marker in message for marker in runtime_unavailable_markers):
            return (
                f"{message} "
                "Do NOT retry browser tool; browser runtime is unavailable. "
                "Check sandbox browser runtime/container health."
            )
        return message

    async def _await_provider_call(self, action: str, coro):
        timeout_s = self._action_timeout_seconds(action)
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except TimeoutError as e:
            raise ValueError(
                f"browser_action_timeout:{action}:{int(timeout_s)}s"
            ) from e

    async def execute_action(
        self,
        *,
        action: str,
        effective_user_id: str,
        provider_name: str | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        profile_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> BrowserActionResult:
        params = params or {}

        if not self._config.browser.enabled:
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="browser_disabled",
                error_message="browser integration is disabled",
            )

        await self._apply_retention_policies(effective_user_id=effective_user_id)

        provider_key = (provider_name or self._config.browser.provider).strip().lower()
        provider = self._providers.get(provider_key)
        if provider is None:
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="invalid_provider",
                error_message=f"unsupported provider: {provider_key}",
            )
        if provider_key == "kernel" and action.startswith("page."):
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="provider_not_supported",
                error_message=(
                    "kernel provider does not currently support page actions; "
                    "use sandbox provider"
                ),
            )
        if (
            provider_key == "sandbox"
            and not self._is_sandbox_runtime()
            and not bool(getattr(provider, "runs_in_sandbox_executor", False))
        ):
            message = (
                "sandbox_runtime_required: browser provider 'sandbox' must run "
                "inside sandbox/container runtime (current process is host runtime)"
            )
            logger.warning(
                "browser_sandbox_runtime_required",
                extra={
                    "browser.action": action,
                    "browser.provider": provider_key,
                    "error.message": message,
                },
            )
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="sandbox_runtime_required",
                error_message=message,
            )

        logger.info(
            "browser_action_started",
            extra={
                "browser.action": action,
                "browser.provider": provider_key,
                "browser.has_session_id": bool(session_id),
                "browser.has_session_name": bool(session_name),
            },
        )

        try:
            if action == "session.start":
                return await self._session_start(
                    provider=provider,
                    provider_key=provider_key,
                    effective_user_id=effective_user_id,
                    session_name=session_name,
                    profile_name=profile_name,
                )
            if action == "session.list":
                return self._session_list(effective_user_id=effective_user_id)
            if action == "session.show":
                return self._session_show(
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider=provider_key,
                )
            if action == "session.close":
                return await self._session_close(
                    provider=provider,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                )
            if action == "session.archive":
                return await self._session_archive(
                    provider=provider,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                )
            if action == "page.goto":
                raw_timeout = params.get("timeout_seconds")
                return await self._page_goto(
                    provider=provider,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                    url=str(params.get("url") or "").strip(),
                    timeout_seconds=self._clamp_timeout_seconds(
                        float(
                            raw_timeout
                            if raw_timeout is not None
                            else self._config.browser.timeout_seconds
                        )
                    ),
                )
            if action == "page.extract":
                return await self._page_extract(
                    provider=provider,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                    mode=str(params.get("mode") or "text").strip().lower(),
                    selector=(
                        str(params["selector"]) if params.get("selector") else None
                    ),
                    max_chars=int(params.get("max_chars") or 3000),
                )
            if action == "page.click":
                return await self._page_click_or_type_or_wait(
                    provider=provider,
                    action=action,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                    params=params,
                )
            if action == "page.type":
                return await self._page_click_or_type_or_wait(
                    provider=provider,
                    action=action,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                    params=params,
                )
            if action == "page.wait_for":
                return await self._page_click_or_type_or_wait(
                    provider=provider,
                    action=action,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                    params=params,
                )
            if action == "page.screenshot":
                return await self._page_screenshot(
                    provider=provider,
                    effective_user_id=effective_user_id,
                    session_id=session_id,
                    session_name=session_name,
                    provider_name=provider_key,
                )
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="unknown_action",
                error_message=f"unknown browser action: {action}",
            )
        except Exception as e:
            message = self._normalize_browser_error_message(str(e))
            logger.warning(
                "browser_action_failed",
                extra={
                    "browser.action": action,
                    "browser.provider": provider_key,
                    "error.message": message,
                },
            )
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="action_failed",
                error_message=message,
            )

    async def _session_start(
        self,
        *,
        provider: BrowserProvider,
        provider_key: str,
        effective_user_id: str,
        session_name: str | None,
        profile_name: str | None,
    ) -> BrowserActionResult:
        session_id = str(uuid.uuid4())
        resolved_name = (session_name or f"session-{session_id[:8]}").strip()
        started = await self._await_provider_call(
            "session.start",
            provider.start_session(
                session_id=session_id,
                profile_name=profile_name,
                scope_key=effective_user_id,
            ),
        )

        now = datetime.now(UTC)
        session = BrowserSession(
            id=session_id,
            name=resolved_name,
            effective_user_id=effective_user_id,
            provider=provider_key,  # type: ignore[arg-type]
            profile_name=profile_name,
            provider_session_id=started.provider_session_id,
            metadata=dict(started.metadata),
            created_at=now,
            updated_at=now,
        )
        self._store.append_session(session)

        if profile_name:
            existing = self._store.get_profile(
                name=profile_name,
                effective_user_id=effective_user_id,
                provider=provider_key,
            )
            if existing is None:
                profile = BrowserProfile(
                    name=profile_name,
                    effective_user_id=effective_user_id,
                    provider=provider_key,  # type: ignore[arg-type]
                    created_at=now,
                    updated_at=now,
                )
                self._store.append_profile(profile)

        logger.info(
            "browser_session_started",
            extra={
                "browser.session_id": session.id,
                "browser.session_name": session.name,
                "browser.provider": session.provider,
                "browser.profile_name": session.profile_name,
            },
        )
        return BrowserActionResult(
            ok=True,
            action="session.start",
            session_id=session.id,
            provider=session.provider,
            data={
                "session_name": session.name,
                "profile_name": session.profile_name,
                "status": session.status,
            },
        )

    def _session_list(self, *, effective_user_id: str) -> BrowserActionResult:
        sessions = self._store.list_sessions(effective_user_id=effective_user_id)
        items = [
            {
                "id": s.id,
                "name": s.name,
                "provider": s.provider,
                "status": s.status,
                "profile_name": s.profile_name,
                "current_url": s.current_url,
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
        return BrowserActionResult(
            ok=True,
            action="session.list",
            data={"sessions": items, "count": len(items)},
        )

    def _session_show(
        self,
        *,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider: str,
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="session.show",
                error_code="session_not_found",
                error_message="session not found",
            )
        return BrowserActionResult(
            ok=True,
            action="session.show",
            session_id=session.id,
            provider=session.provider,
            page_url=session.current_url,
            data={
                "id": session.id,
                "name": session.name,
                "status": session.status,
                "profile_name": session.profile_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
            },
        )

    async def _session_close(
        self,
        *,
        provider: BrowserProvider,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="session.close",
                error_code="session_not_found",
                error_message="session not found",
            )
        await self._await_provider_call(
            "session.close",
            provider.close_session(provider_session_id=session.provider_session_id),
        )

        closed = replace(session, status="closed", updated_at=datetime.now(UTC))
        self._store.append_session(closed)
        logger.info("browser_session_closed", extra={"browser.session_id": session.id})
        return BrowserActionResult(
            ok=True,
            action="session.close",
            session_id=session.id,
            provider=session.provider,
            data={"status": "closed"},
        )

    async def _session_archive(
        self,
        *,
        provider: BrowserProvider,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
            include_archived=True,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="session.archive",
                error_code="session_not_found",
                error_message="session not found",
            )

        if session.status == "active":
            await self._await_provider_call(
                "session.close",
                provider.close_session(provider_session_id=session.provider_session_id),
            )

        archived = replace(
            session,
            status="archived",
            updated_at=datetime.now(UTC),
        )
        self._store.append_session(archived)
        logger.info(
            "browser_session_archived",
            extra={
                "browser.session_id": session.id,
                "browser.session_name": session.name,
            },
        )
        return BrowserActionResult(
            ok=True,
            action="session.archive",
            session_id=session.id,
            provider=session.provider,
            data={"status": "archived"},
        )

    async def _page_goto(
        self,
        *,
        provider: BrowserProvider,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
        url: str,
        timeout_seconds: float,
    ) -> BrowserActionResult:
        if not url:
            return BrowserActionResult(
                ok=False,
                action="page.goto",
                error_code="missing_url",
                error_message="url is required",
            )
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="page.goto",
                error_code="session_not_found",
                error_message="session not found",
            )

        result: ProviderGotoResult = await self._await_provider_call(
            "page.goto",
            provider.goto(
                provider_session_id=session.provider_session_id,
                url=url,
                timeout_seconds=max(1.0, timeout_seconds),
            ),
        )

        artifact_refs: list[str] = []
        metadata = dict(session.metadata)
        if result.html:
            html_path = self._write_artifact(
                session_id=session.id,
                suffix=".html",
                content=result.html.encode("utf-8", errors="replace"),
            )
            metadata["last_html_path"] = str(html_path)
            artifact_refs.append(str(html_path))
        if result.title:
            metadata["last_title"] = result.title

        updated = replace(
            session,
            current_url=result.url,
            metadata=metadata,
            updated_at=datetime.now(UTC),
        )
        self._store.append_session(updated)

        logger.info(
            "browser_action_succeeded",
            extra={
                "browser.action": "page.goto",
                "browser.session_id": session.id,
                "browser.provider": session.provider,
            },
        )
        return BrowserActionResult(
            ok=True,
            action="page.goto",
            session_id=session.id,
            provider=session.provider,
            page_url=result.url,
            data={"title": result.title},
            artifact_refs=artifact_refs,
        )

    async def _page_extract(
        self,
        *,
        provider: BrowserProvider,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="page.extract",
                error_code="session_not_found",
                error_message="session not found",
            )

        html = self._read_last_html(session)
        extract = await self._await_provider_call(
            "page.extract",
            provider.extract(
                provider_session_id=session.provider_session_id,
                html=html,
                mode=mode,
                selector=selector,
                max_chars=max(1, max_chars),
            ),
        )

        logger.info(
            "browser_action_succeeded",
            extra={
                "browser.action": "page.extract",
                "browser.session_id": session.id,
                "browser.provider": session.provider,
            },
        )
        return BrowserActionResult(
            ok=True,
            action="page.extract",
            session_id=session.id,
            provider=session.provider,
            page_url=session.current_url,
            data=extract.data,
        )

    async def _page_click_or_type_or_wait(
        self,
        *,
        provider: BrowserProvider,
        action: str,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
        params: dict[str, Any],
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="session_not_found",
                error_message="session not found",
            )

        selector = str(params.get("selector") or "").strip()
        if not selector:
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="missing_selector",
                error_message="selector is required",
            )

        if action == "page.click":
            await self._await_provider_call(
                action,
                provider.click(
                    provider_session_id=session.provider_session_id,
                    selector=selector,
                ),
            )
        elif action == "page.type":
            await self._await_provider_call(
                action,
                provider.type(
                    provider_session_id=session.provider_session_id,
                    selector=selector,
                    text=str(params.get("text") or ""),
                    clear_first=bool(params.get("clear_first", True)),
                ),
            )
        else:
            raw_timeout = params.get("timeout_seconds")
            await self._await_provider_call(
                action,
                provider.wait_for(
                    provider_session_id=session.provider_session_id,
                    selector=selector,
                    timeout_seconds=self._clamp_timeout_seconds(
                        float(
                            raw_timeout
                            if raw_timeout is not None
                            else self._config.browser.timeout_seconds
                        )
                    ),
                ),
            )

        logger.info(
            "browser_action_succeeded",
            extra={
                "browser.action": action,
                "browser.session_id": session.id,
                "browser.provider": session.provider,
            },
        )
        return BrowserActionResult(
            ok=True,
            action=action,
            session_id=session.id,
            provider=session.provider,
            page_url=session.current_url,
        )

    async def _page_screenshot(
        self,
        *,
        provider: BrowserProvider,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
    ) -> BrowserActionResult:
        session = self._resolve_session(
            effective_user_id=effective_user_id,
            session_id=session_id,
            session_name=session_name,
            provider_name=provider_name,
        )
        if session is None:
            return BrowserActionResult(
                ok=False,
                action="page.screenshot",
                error_code="session_not_found",
                error_message="session not found",
            )

        shot = await self._await_provider_call(
            "page.screenshot",
            provider.screenshot(provider_session_id=session.provider_session_id),
        )
        ext = ".png" if shot.mime_type == "image/png" else ".bin"
        path = self._write_artifact(
            session_id=session.id, suffix=ext, content=shot.image_bytes
        )

        logger.info(
            "browser_action_succeeded",
            extra={
                "browser.action": "page.screenshot",
                "browser.session_id": session.id,
                "browser.provider": session.provider,
            },
        )
        return BrowserActionResult(
            ok=True,
            action="page.screenshot",
            session_id=session.id,
            provider=session.provider,
            page_url=session.current_url,
            artifact_refs=[str(path)],
        )

    def _resolve_session(
        self,
        *,
        effective_user_id: str,
        session_id: str | None,
        session_name: str | None,
        provider_name: str,
        include_archived: bool = False,
    ) -> BrowserSession | None:
        if session_id:
            session = self._store.get_session(session_id)
            if session and session.effective_user_id == effective_user_id:
                if session.provider != provider_name:
                    return None
                if include_archived or session.status != "archived":
                    return session
            return None
        if session_name:
            return self._store.get_session_by_name(
                name=session_name,
                effective_user_id=effective_user_id,
                provider=provider_name,
                include_archived=include_archived,
            )
        sessions = self._store.list_sessions(
            effective_user_id=effective_user_id,
            include_archived=include_archived,
        )
        if not sessions:
            return None
        provider_sessions = [s for s in sessions if s.provider == provider_name]
        return provider_sessions[0] if provider_sessions else None

    async def _apply_retention_policies(self, *, effective_user_id: str) -> None:
        """Apply browser session/artifact retention best-effort."""
        await self._expire_stale_sessions(effective_user_id=effective_user_id)
        self._prune_artifacts()

    async def _expire_stale_sessions(self, *, effective_user_id: str) -> None:
        max_age_minutes = self._config.browser.max_session_minutes
        if max_age_minutes <= 0:
            return

        cutoff = datetime.now(UTC) - timedelta(minutes=max_age_minutes)
        sessions = self._store.list_sessions(
            effective_user_id=effective_user_id,
            include_archived=False,
        )
        for session in sessions:
            if session.status != "active":
                continue
            if session.updated_at >= cutoff:
                continue
            provider = self._providers.get(session.provider)
            if provider is not None:
                try:
                    await self._await_provider_call(
                        "session.close",
                        provider.close_session(
                            provider_session_id=session.provider_session_id
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        "browser_session_close_failed",
                        extra={
                            "browser.session_id": session.id,
                            "browser.provider": session.provider,
                            "error.message": str(e),
                        },
                    )
            expired = replace(
                session,
                status="closed",
                last_error="session_expired",
                updated_at=datetime.now(UTC),
            )
            self._store.append_session(expired)
            logger.info(
                "browser_session_expired",
                extra={
                    "browser.session_id": session.id,
                    "browser.session_name": session.name,
                    "browser.provider": session.provider,
                },
            )

    def _prune_artifacts(self) -> None:
        retention_days = self._config.browser.artifacts_retention_days
        if retention_days < 0:
            return

        artifacts_dir = self._store.artifacts_dir
        if not artifacts_dir.exists():
            return

        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        removed_files = 0
        for path in artifacts_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                if modified_at < cutoff:
                    path.unlink()
                    removed_files += 1
            except OSError:
                continue

        removed_dirs = 0
        # Cleanup empty session artifact directories after file pruning
        for path in sorted(artifacts_dir.rglob("*"), reverse=True):
            if not path.is_dir():
                continue
            try:
                if any(path.iterdir()):
                    continue
                shutil.rmtree(path)
                removed_dirs += 1
            except OSError:
                continue

        if removed_files or removed_dirs:
            logger.info(
                "browser_artifacts_pruned",
                extra={
                    "browser.removed_file_count": removed_files,
                    "browser.removed_dir_count": removed_dirs,
                    "browser.retention_days": retention_days,
                },
            )

    def _session_artifacts_dir(self, session_id: str) -> Path:
        path = self._store.artifacts_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_artifact(self, *, session_id: str, suffix: str, content: bytes) -> Path:
        path = self._session_artifacts_dir(session_id) / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + suffix
        )
        path.write_bytes(content)
        logger.info("browser_artifact_written", extra={"file.path": str(path)})
        return path

    def _read_last_html(self, session: BrowserSession) -> str | None:
        path_value = session.metadata.get("last_html_path")
        if not isinstance(path_value, str) or not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        try:
            return path.read_text(errors="replace")
        except OSError:
            return None


def create_browser_manager(
    config: AshConfig,
    *,
    sandbox_executor: SandboxExecutor | None = None,
) -> BrowserManager:
    """Create a fully-wired browser manager."""
    # Runtime boundary contract: specs/browser.md
    # Sandbox provider execution must remain sandbox/container-scoped.
    browser_cfg = config.browser
    base_dir = (
        browser_cfg.state_dir.expanduser()
        if browser_cfg.state_dir
        else get_browser_path()
    )
    store = BrowserStore(base_dir)

    kernel_key = None
    if config.browser.kernel and config.browser.kernel.api_key:
        kernel_key = config.browser.kernel.api_key.get_secret_value()
    if not kernel_key:
        kernel_key = os.environ.get("KERNEL_API_KEY")

    providers: dict[str, BrowserProvider] = {}
    # Provider exposure is single-source-of-truth from config.browser.provider.
    if browser_cfg.provider == "sandbox":
        providers["sandbox"] = SandboxBrowserProvider(
            headless=config.browser.sandbox.headless,
            browser_channel=config.browser.sandbox.browser_channel,
            viewport_width=config.browser.default_viewport_width,
            viewport_height=config.browser.default_viewport_height,
            executor=sandbox_executor,
            runtime_mode=config.browser.sandbox.runtime_mode,
            container_image=config.browser.sandbox.container_image,
            container_name_prefix=config.browser.sandbox.container_name_prefix,
            runtime_restart_attempts=config.browser.sandbox.runtime_restart_attempts,
        )
    else:
        providers["kernel"] = KernelBrowserProvider(
            api_key=kernel_key,
            base_url=config.browser.kernel.base_url
            if config.browser.kernel
            else "https://api.kernel.sh",
            project_id=config.browser.kernel.project_id
            if config.browser.kernel
            else None,
        )
    return BrowserManager(config=config, store=store, providers=providers)


def format_browser_result(result: BrowserActionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=True, indent=2)
