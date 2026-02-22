"""Sandbox security verification tests.

These tests verify the sandbox correctly restricts dangerous operations
while allowing legitimate use cases.

Requirements:
    - Docker running
    - Sandbox image built: `ash sandbox build`

Run with:
    pytest tests/test_sandbox_verify.py -v

Skip network tests:
    pytest tests/test_sandbox_verify.py -v -m "not network"
"""

import subprocess

import pytest

from ash.sandbox.verify import (
    VERIFICATION_TESTS,
    SandboxVerifier,
    VerificationResult,
    VerificationTest,
)


def _docker_available() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _sandbox_image_exists() -> bool:
    """Check if sandbox image is built."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "ash-sandbox:latest"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return False


# Skip all tests if Docker not available or image not built
pytestmark = [
    pytest.mark.skipif(not _docker_available(), reason="Docker not available"),
    pytest.mark.skipif(not _sandbox_image_exists(), reason="Sandbox image not built"),
]


@pytest.fixture
async def verifier():
    """Create a sandbox verifier with network enabled."""
    v = SandboxVerifier(network_enabled=True)
    yield v
    await v.cleanup()


@pytest.fixture
async def verifier_no_network():
    """Create a sandbox verifier without network."""
    v = SandboxVerifier(network_enabled=False)
    yield v
    await v.cleanup()


# =============================================================================
# Security Tests
# =============================================================================


class TestSandboxSecurity:
    """Security boundary tests."""

    @pytest.mark.asyncio
    async def test_user_is_sandbox(self, verifier: SandboxVerifier):
        """Commands run as unprivileged 'sandbox' user."""
        test = _get_test("user_is_sandbox")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_user_not_root(self, verifier: SandboxVerifier):
        """User is not root (UID != 0)."""
        test = _get_test("user_not_root")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_sudo_blocked(self, verifier: SandboxVerifier):
        """sudo command is unavailable or blocked."""
        test = _get_test("sudo_blocked")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_su_blocked(self, verifier: SandboxVerifier):
        """su command is blocked."""
        test = _get_test("su_blocked")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_etc_readonly(self, verifier: SandboxVerifier):
        """/etc is read-only."""
        test = _get_test("etc_readonly")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_usr_readonly(self, verifier: SandboxVerifier):
        """/usr is read-only."""
        test = _get_test("usr_readonly")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_bin_readonly(self, verifier: SandboxVerifier):
        """/bin is read-only."""
        test = _get_test("bin_readonly")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_root_home_inaccessible(self, verifier: SandboxVerifier):
        """/root is not accessible."""
        test = _get_test("root_home_inaccessible")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message


# =============================================================================
# Resource Limit Tests
# =============================================================================


class TestSandboxResources:
    """Resource limit tests."""

    @pytest.mark.asyncio
    async def test_timeout_enforced(self, verifier: SandboxVerifier):
        """Commands timeout after limit."""
        test = _get_test("timeout_enforced")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_tmp_writable(self, verifier: SandboxVerifier):
        """/tmp is writable (tmpfs)."""  # noqa: S108
        test = _get_test("tmp_writable")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_home_writable(self, verifier: SandboxVerifier):
        """/home/sandbox is writable (tmpfs)."""
        test = _get_test("home_writable")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_tmp_noexec(self, verifier: SandboxVerifier):
        """/tmp has noexec (can't run scripts directly)."""  # noqa: S108
        test = _get_test("tmp_noexec")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message


# =============================================================================
# Functional Tests
# =============================================================================


class TestSandboxFunctional:
    """Functional tests - commands that should work."""

    @pytest.mark.asyncio
    async def test_workspace_exists(self, verifier: SandboxVerifier):
        """Workspace directory exists."""
        test = _get_test("workspace_exists")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_python_available(self, verifier: SandboxVerifier):
        """Python is available."""
        test = _get_test("python_available")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_python_execution(self, verifier: SandboxVerifier):
        """Python can execute code."""
        test = _get_test("python_execution")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_bash_available(self, verifier: SandboxVerifier):
        """Bash is available."""
        test = _get_test("bash_available")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_git_available(self, verifier: SandboxVerifier):
        """Git is available."""
        test = _get_test("git_available")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_jq_available(self, verifier: SandboxVerifier):
        """jq is available for JSON processing."""
        test = _get_test("jq_available")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_curl_available(self, verifier: SandboxVerifier):
        """curl is available."""
        test = _get_test("curl_available")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message


# =============================================================================
# Network Tests
# =============================================================================


@pytest.mark.network
class TestSandboxNetwork:
    """Network tests - require external connectivity."""

    @pytest.mark.asyncio
    async def test_dns_resolution(self, verifier: SandboxVerifier):
        """DNS resolution works."""
        test = _get_test("dns_resolution")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_https_request(self, verifier: SandboxVerifier):
        """HTTPS requests work."""
        test = _get_test("https_request")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_http_request(self, verifier: SandboxVerifier):
        """HTTP requests work."""
        test = _get_test("http_request")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSandboxEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_special_characters(self, verifier: SandboxVerifier):
        """Special characters handled correctly."""
        test = _get_test("special_characters")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_multiline_output(self, verifier: SandboxVerifier):
        """Multiline output works."""
        test = _get_test("multiline_output")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_exit_code_preserved(self, verifier: SandboxVerifier):
        """Exit codes are preserved."""
        test = _get_test("exit_code_preserved")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_stderr_captured(self, verifier: SandboxVerifier):
        """Stderr is captured."""
        test = _get_test("stderr_captured")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_empty_output(self, verifier: SandboxVerifier):
        """Empty output handled."""
        test = _get_test("empty_output")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message

    @pytest.mark.asyncio
    async def test_large_output_truncated(self, verifier: SandboxVerifier):
        """Large output is handled."""
        test = _get_test("large_output_truncated")
        result = await verifier.run_test(test)
        assert result.result == VerificationResult.PASS, result.message


# =============================================================================
# Helpers
# =============================================================================


def _get_test(name: str) -> VerificationTest:
    """Get a verification test by name."""
    for test in VERIFICATION_TESTS:
        if test.name == name:
            return test
    raise ValueError(f"Test not found: {name}")
