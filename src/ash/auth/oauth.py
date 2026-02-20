"""OpenAI Codex OAuth PKCE flow.

Ported from pi-mono/packages/ai/src/utils/oauth/openai-codex.ts.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import webbrowser
from asyncio import Event
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"

SUCCESS_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authentication successful</title>
</head>
<body>
  <p>Authentication successful. Return to your terminal to continue.</p>
</body>
</html>"""

CALLBACK_TIMEOUT_SECONDS = 120


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and SHA-256 challenge.

    Returns:
        Tuple of (verifier, challenge).
    """
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url_encode(verifier_bytes)

    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url_encode(digest)

    return verifier, challenge


def build_authorization_url(
    verifier_challenge: str,
    state: str,
    originator: str = "ash",
) -> str:
    """Build the OpenAI authorization URL with PKCE params."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": verifier_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_authorization_code(
    code: str,
    verifier: str,
) -> dict[str, str | float]:
    """Exchange authorization code for tokens.

    Returns:
        Dict with keys: access, refresh, expires (timestamp in seconds).

    Raises:
        RuntimeError: If token exchange fails.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        logger.error(
            "Token exchange failed: %d %s", response.status_code, response.text
        )
        raise RuntimeError(f"Token exchange failed: {response.status_code}")

    data = response.json()
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    expires_in = data.get("expires_in")

    if not access_token or not refresh_token or not isinstance(expires_in, int | float):
        raise RuntimeError(
            f"Token response missing required fields: {list(data.keys())}"
        )

    import time

    return {
        "access": access_token,
        "refresh": refresh_token,
        "expires": time.time() + float(expires_in),
    }


async def refresh_access_token(refresh_token: str) -> dict[str, str | float]:
    """Refresh an access token using a refresh token.

    Returns:
        Dict with keys: access, refresh, expires (timestamp in seconds).

    Raises:
        RuntimeError: If token refresh fails.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        logger.error("Token refresh failed: %d %s", response.status_code, response.text)
        raise RuntimeError(f"Token refresh failed: {response.status_code}")

    data = response.json()
    access_token = data.get("access_token")
    new_refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")

    if not access_token or not new_refresh or not isinstance(expires_in, int | float):
        raise RuntimeError(
            f"Token refresh response missing fields: {list(data.keys())}"
        )

    import time

    return {
        "access": access_token,
        "refresh": new_refresh,
        "expires": time.time() + float(expires_in),
    }


def extract_account_id(access_token: str) -> str | None:
    """Extract chatgpt_account_id from a JWT access token.

    Decodes the JWT payload (base64url, no verification) and reads the
    ``https://api.openai.com/auth`` claim.

    Returns:
        The account ID string, or None if not found.
    """
    parts = access_token.split(".")
    if len(parts) != 3:
        return None

    payload_b64 = parts[1]
    # Add padding if needed
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding

    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_bytes)
    except Exception:
        return None

    auth_claim = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth_claim, dict):
        return None

    account_id = auth_claim.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    return None


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for the OAuth callback."""

    # Set by the factory
    expected_state: str = ""
    received_code: str | None = None
    code_event: Event | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)

        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]

        if state != self.expected_state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return

        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(SUCCESS_HTML.encode())

        _CallbackHandler.received_code = code
        if _CallbackHandler.code_event and _CallbackHandler._loop:
            _CallbackHandler._loop.call_soon_threadsafe(_CallbackHandler.code_event.set)

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default HTTP server logging
        pass


async def _start_callback_server(
    state: str,
) -> tuple[HTTPServer | None, str | None]:
    """Start local callback server and wait for the authorization code.

    Returns:
        Tuple of (server, authorization_code). Server may be None if bind fails.
    """
    code_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    _CallbackHandler.expected_state = state
    _CallbackHandler.received_code = None
    _CallbackHandler.code_event = code_event
    _CallbackHandler._loop = loop

    try:
        server = HTTPServer(("127.0.0.1", 1455), _CallbackHandler)
    except OSError as e:
        logger.warning("Failed to bind callback server on port 1455: %s", e)
        return None, None

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        await asyncio.wait_for(code_event.wait(), timeout=CALLBACK_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning("OAuth callback timed out after %ds", CALLBACK_TIMEOUT_SECONDS)
        return server, None
    finally:
        server.shutdown()
        thread.join(timeout=2)

    return server, _CallbackHandler.received_code


async def login_openai_codex() -> dict[str, str | float]:
    """Run the full OpenAI Codex OAuth login flow.

    Opens the browser for authentication, starts a local callback server,
    and exchanges the authorization code for tokens.

    Returns:
        Dict with keys: access (str), refresh (str), expires (float), account_id (str).

    Raises:
        RuntimeError: If any step of the flow fails.
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)

    url = build_authorization_url(challenge, state)

    # Start callback server and open browser
    print(
        f"\nOpening browser for authentication...\nIf it doesn't open, visit:\n{url}\n"
    )
    webbrowser.open(url)

    _, code = await _start_callback_server(state)

    if not code:
        raise RuntimeError(
            "Did not receive authorization code. "
            "Please try again or check your browser."
        )

    tokens = await exchange_authorization_code(code, verifier)

    access = str(tokens["access"])
    account_id = extract_account_id(access)
    if not account_id:
        raise RuntimeError("Failed to extract account_id from access token")

    return {
        "access": access,
        "refresh": str(tokens["refresh"]),
        "expires": float(tokens["expires"]),
        "account_id": account_id,
    }
