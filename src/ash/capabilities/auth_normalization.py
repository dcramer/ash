"""Central auth-complete input normalization for capability auth flows."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


class AuthNormalizationError(ValueError):
    """Normalization error with stable capability auth error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class NormalizedAuthCompletion:
    """Canonical auth completion payload used by capability providers."""

    authorization_code: str
    raw_callback_url: str | None
    state: str | None


def normalize_auth_completion(
    *,
    callback_url: str | None,
    code: str | None,
    expected_state: str | None,
) -> NormalizedAuthCompletion:
    """Normalize callback URL / code inputs into one authorization code."""
    normalized_code = _optional_text(code)
    normalized_callback_url = _optional_text(callback_url)
    callback_code: str | None = None
    callback_state: str | None = None

    if normalized_callback_url is not None:
        callback_code, callback_state = _parse_callback_url(normalized_callback_url)

    if (
        normalized_code is not None
        and callback_code is not None
        and normalized_code != callback_code
    ):
        raise AuthNormalizationError(
            "capability_auth_code_conflict",
            "code does not match callback_url code",
        )

    if (
        expected_state is not None
        and callback_state is not None
        and callback_state != expected_state
    ):
        raise AuthNormalizationError(
            "capability_auth_state_mismatch",
            "callback_url state does not match auth flow",
        )

    authorization_code = normalized_code or callback_code
    if authorization_code is None:
        raise AuthNormalizationError(
            "capability_auth_code_missing",
            "either code or callback_url with code is required",
        )

    return NormalizedAuthCompletion(
        authorization_code=authorization_code,
        raw_callback_url=normalized_callback_url,
        state=callback_state,
    )


def _parse_callback_url(callback_url: str) -> tuple[str, str | None]:
    parsed = urlparse(callback_url)
    if not parsed.scheme or not parsed.netloc:
        raise AuthNormalizationError(
            "capability_auth_callback_invalid",
            "callback_url is not a valid URL",
        )

    query = parse_qs(parsed.query)
    code = _optional_text((query.get("code") or [None])[0])
    if code is None:
        raise AuthNormalizationError(
            "capability_auth_code_missing",
            "callback_url missing code query parameter",
        )
    state = _optional_text((query.get("state") or [None])[0])
    return code, state


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
