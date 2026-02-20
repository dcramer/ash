"""Tests for OAuth PKCE flow."""

import base64
import hashlib
import json

import pytest

from ash.auth.oauth import (
    _parse_callback_url,
    build_authorization_url,
    extract_account_id,
    generate_pkce,
)


class TestParseCallbackURL:
    def test_extracts_code(self):
        url = "http://localhost:1455/auth/callback?code=abc123&state=mystate"
        assert _parse_callback_url(url, "mystate") == "abc123"

    def test_strips_whitespace(self):
        url = "  http://localhost:1455/auth/callback?code=abc&state=s  \n"
        assert _parse_callback_url(url, "s") == "abc"

    def test_state_mismatch_raises(self):
        url = "http://localhost:1455/auth/callback?code=abc&state=wrong"
        with pytest.raises(RuntimeError, match="State mismatch"):
            _parse_callback_url(url, "expected")

    def test_missing_code_raises(self):
        url = "http://localhost:1455/auth/callback?state=s"
        with pytest.raises(RuntimeError, match="No authorization code"):
            _parse_callback_url(url, "s")


class TestGeneratePKCE:
    def test_returns_verifier_and_challenge(self):
        verifier, challenge = generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 0
        assert len(challenge) > 0

    def test_challenge_is_sha256_of_verifier(self):
        verifier, challenge = generate_pkce()
        # Recompute challenge from verifier
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_verifiers_are_unique(self):
        v1, _ = generate_pkce()
        v2, _ = generate_pkce()
        assert v1 != v2

    def test_verifier_is_base64url(self):
        verifier, _ = generate_pkce()
        # base64url chars only, no padding
        assert "+" not in verifier
        assert "/" not in verifier
        assert "=" not in verifier


class TestBuildAuthorizationURL:
    def test_contains_required_params(self):
        url = build_authorization_url("test-challenge", "test-state")
        assert "response_type=code" in url
        assert "client_id=" in url
        assert "redirect_uri=" in url
        assert "code_challenge=test-challenge" in url
        assert "code_challenge_method=S256" in url
        assert "state=test-state" in url
        assert "scope=" in url

    def test_contains_originator(self):
        url = build_authorization_url("c", "s", originator="ash")
        assert "originator=ash" in url

    def test_custom_originator(self):
        url = build_authorization_url("c", "s", originator="custom")
        assert "originator=custom" in url

    def test_starts_with_authorize_url(self):
        url = build_authorization_url("c", "s")
        assert url.startswith("https://auth.openai.com/oauth/authorize?")


class TestExtractAccountId:
    def _make_jwt(self, payload: dict) -> str:
        """Create a fake JWT with the given payload."""
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        body = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        )
        sig = base64.urlsafe_b64encode(b"fake-sig").rstrip(b"=").decode()
        return f"{header}.{body}.{sig}"

    def test_extracts_account_id(self):
        token = self._make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123abc"}}
        )
        assert extract_account_id(token) == "acct_123abc"

    def test_returns_none_for_missing_claim(self):
        token = self._make_jwt({"sub": "user123"})
        assert extract_account_id(token) is None

    def test_returns_none_for_empty_account_id(self):
        token = self._make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": ""}}
        )
        assert extract_account_id(token) is None

    def test_returns_none_for_invalid_token(self):
        assert extract_account_id("not-a-jwt") is None

    def test_returns_none_for_non_dict_claim(self):
        token = self._make_jwt({"https://api.openai.com/auth": "not-a-dict"})
        assert extract_account_id(token) is None
