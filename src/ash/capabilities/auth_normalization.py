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
    normalized_code: str | None = None
    normalized_callback_url = _optional_text(callback_url)
    callback_code: str | None = None
    callback_state: str | None = None

    if code is not None:
        (
            normalized_code,
            inferred_callback_url,
            inferred_state,
        ) = _normalize_code_input(code)
        if normalized_callback_url is None and inferred_callback_url is not None:
            normalized_callback_url = inferred_callback_url
        if callback_state is None and inferred_state is not None:
            callback_state = inferred_state

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


def _normalize_code_input(code: str) -> tuple[str | None, str | None, str | None]:
    text = _optional_text(code)
    if text is None:
        return None, None, None

    cleaned = _trim_wrapper_punctuation(text)
    compact = "".join(cleaned.split())

    try:
        code_from_url, state_from_url = _parse_callback_url(compact)
    except AuthNormalizationError:
        pass
    else:
        return code_from_url, compact, state_from_url

    code_from_query, state_from_query = _extract_code_from_query_fragment(compact)
    if code_from_query is not None:
        return code_from_query, None, state_from_query

    return compact, None, None


def _extract_code_from_query_fragment(text: str) -> tuple[str | None, str | None]:
    fragment = text
    if "?" in fragment:
        fragment = fragment.split("?", 1)[1]

    if "code=" not in fragment:
        return None, None

    query = parse_qs(fragment, keep_blank_values=True)
    code = _optional_text((query.get("code") or [None])[0])
    state = _optional_text((query.get("state") or [None])[0])
    return code, state


def _trim_wrapper_punctuation(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2:
        pairs = (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("`", "`"))
        changed = True
        while changed and len(cleaned) >= 2:
            changed = False
            for left, right in pairs:
                if cleaned.startswith(left) and cleaned.endswith(right):
                    cleaned = cleaned[1:-1].strip()
                    changed = True
                    break
    return cleaned.rstrip(".,;")
