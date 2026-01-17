"""Tests for logging configuration and utilities."""

import os
import time

from ash.logging import (
    DEFAULT_REDACT_PATTERNS,
    SecretRedactor,
    prune_old_logs,
)


class TestSecretRedactor:
    """Tests for SecretRedactor class."""

    def test_redacts_anthropic_api_key(self):
        redactor = SecretRedactor()
        text = "Using API key sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        assert "sk-a" in result
        assert "3456" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_redacts_openai_api_key(self):
        redactor = SecretRedactor()
        text = "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        # Should preserve OPENAI_API_KEY= prefix
        assert "OPENAI_API_KEY=" in result
        # Should show partial token (first 4 and last 4 chars)
        assert "sk-p" in result
        assert "3456" in result
        # Middle should be masked
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_redacts_github_pat(self):
        redactor = SecretRedactor()
        text = "Token: ghp_abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        assert "ghp_" in result
        assert "3456" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_redacts_github_fine_grained_pat(self):
        redactor = SecretRedactor()
        text = "github_pat_abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        assert "gith" in result
        assert "3456" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_redacts_slack_token(self):
        redactor = SecretRedactor()
        text = "xoxb-123456789012-abcdefghijk"
        result = redactor.redact(text)
        assert "xoxb" in result
        assert "hijk" in result
        assert "123456789012" not in result

    def test_redacts_telegram_bot_token(self):
        redactor = SecretRedactor()
        text = "Bot token: 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ1234567890"
        result = redactor.redact(text)
        assert "1234" in result
        assert "7890" in result
        # Middle part should be redacted
        assert "ABCdefGHIjklMNOpqrSTUvwxYZ" not in result

    def test_redacts_env_style_assignments(self):
        redactor = SecretRedactor()
        text = "API_KEY=verysecretvalue123456"
        result = redactor.redact(text)
        assert "API_KEY=" in result
        assert "verysecretvalue123456" not in result
        # Should show partial
        assert "very" in result
        assert "3456" in result

    def test_redacts_bearer_token(self):
        redactor = SecretRedactor()
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redactor.redact(text)
        assert "Bearer" in result
        assert "eyJh" in result
        assert "VCJ9" in result
        assert "IUzI1NiIsInR5cCI6IkpXV" not in result

    def test_redacts_pem_private_key(self):
        redactor = SecretRedactor()
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJ...
more secret key data here
-----END RSA PRIVATE KEY-----"""
        result = redactor.redact(text)
        assert "-----BEGIN RSA PRIVATE KEY-----" in result
        assert "-----END RSA PRIVATE KEY-----" in result
        assert "...redacted..." in result
        assert "MIIEpAIBAAKCAQEA0Z3VS5JJ" not in result

    def test_preserves_non_secrets(self):
        redactor = SecretRedactor()
        text = "This is a normal log message with no secrets"
        result = redactor.redact(text)
        assert result == text

    def test_preserves_short_values(self):
        redactor = SecretRedactor()
        # Short values that look like secrets but aren't
        text = "API_KEY=short"
        result = redactor.redact(text)
        # 'short' is less than 8 chars, so pattern won't match
        assert result == text

    def test_disabled_redactor_passes_through(self):
        redactor = SecretRedactor(enabled=False)
        text = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
        result = redactor.redact(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        redactor = SecretRedactor()
        assert redactor.redact("") == ""

    def test_multiple_secrets_in_same_text(self):
        redactor = SecretRedactor()
        text = "API_KEY=secret123456789 and ghp_abcdefghijklmnopqrstuvwxyz"
        result = redactor.redact(text)
        # Both should be redacted
        assert "secret123456789" not in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result
        # But partial info preserved
        assert "secr" in result or "API_KEY=" in result
        assert "ghp_" in result

    def test_redacts_password_assignment(self):
        redactor = SecretRedactor()
        text = "DATABASE_PASSWORD=mysupersecretpassword123"
        result = redactor.redact(text)
        assert "DATABASE_PASSWORD=" in result
        assert "mysupersecretpassword123" not in result


class TestPruneOldLogs:
    """Tests for prune_old_logs function."""

    def test_deletes_old_files(self, tmp_path):
        # Create old log file (backdate mtime)
        old_log = tmp_path / "2024-01-01.jsonl"
        old_log.write_text('{"test": "old"}\n')
        # Set mtime to 10 days ago
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(old_log, (old_time, old_time))

        # Create recent log file
        recent_log = tmp_path / "2024-01-10.jsonl"
        recent_log.write_text('{"test": "recent"}\n')

        deleted = prune_old_logs(tmp_path, retention_days=7)

        assert deleted == 1
        assert not old_log.exists()
        assert recent_log.exists()

    def test_keeps_recent_files(self, tmp_path):
        # Create log file from 3 days ago
        recent_log = tmp_path / "recent.jsonl"
        recent_log.write_text('{"test": "recent"}\n')
        recent_time = time.time() - (3 * 24 * 60 * 60)
        os.utime(recent_log, (recent_time, recent_time))

        deleted = prune_old_logs(tmp_path, retention_days=7)

        assert deleted == 0
        assert recent_log.exists()

    def test_ignores_non_jsonl_files(self, tmp_path):
        # Create old non-jsonl file
        old_txt = tmp_path / "old.txt"
        old_txt.write_text("old text")
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(old_txt, (old_time, old_time))

        deleted = prune_old_logs(tmp_path, retention_days=7)

        assert deleted == 0
        assert old_txt.exists()

    def test_handles_nonexistent_directory(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        deleted = prune_old_logs(nonexistent)
        assert deleted == 0

    def test_handles_empty_directory(self, tmp_path):
        deleted = prune_old_logs(tmp_path)
        assert deleted == 0

    def test_custom_retention_period(self, tmp_path):
        # Create log file from 2 days ago
        log_file = tmp_path / "test.jsonl"
        log_file.write_text('{"test": true}\n')
        old_time = time.time() - (2 * 24 * 60 * 60)
        os.utime(log_file, (old_time, old_time))

        # With 1 day retention, should be deleted
        deleted = prune_old_logs(tmp_path, retention_days=1)
        assert deleted == 1
        assert not log_file.exists()

    def test_ignores_directories(self, tmp_path):
        # Create old subdirectory
        old_dir = tmp_path / "old_subdir.jsonl"  # Named like a log file
        old_dir.mkdir()

        deleted = prune_old_logs(tmp_path, retention_days=7)

        assert deleted == 0
        assert old_dir.exists()


class TestDefaultRedactPatterns:
    """Tests for default redaction patterns."""

    def test_all_patterns_compile(self):
        import re

        for pattern in DEFAULT_REDACT_PATTERNS:
            # Should not raise
            re.compile(pattern, re.IGNORECASE)


class TestLogContext:
    """Tests for log_context context manager and related functions."""

    def test_short_id_with_none(self):
        from ash.logging import _short_id

        assert _short_id(None) == ""

    def test_short_id_with_empty_string(self):
        from ash.logging import _short_id

        assert _short_id("") == ""

    def test_short_id_with_telegram_session_key(self):
        from ash.logging import _short_id

        # Should extract chat part from telegram session key
        assert _short_id("telegram_-542863895_1234") == "-5428638"

    def test_short_id_with_discord_session_key(self):
        from ash.logging import _short_id

        assert _short_id("discord_12345678_user") == "12345678"

    def test_short_id_with_slack_session_key(self):
        from ash.logging import _short_id

        assert _short_id("slack_C01234_U56789") == "C01234"

    def test_short_id_with_regular_string(self):
        from ash.logging import _short_id

        assert _short_id("some_session_id_here") == "some_ses"

    def test_short_id_with_custom_length(self):
        from ash.logging import _short_id

        assert _short_id("telegram_-542863895_1234", max_len=5) == "-5428"

    def test_log_context_sets_contextvars(self):
        from ash.logging import _log_chat_id, _log_session_id, log_context

        # Before context
        assert _log_chat_id.get() is None
        assert _log_session_id.get() is None

        with log_context(chat_id="chat123", session_id="session456"):
            assert _log_chat_id.get() == "chat123"
            assert _log_session_id.get() == "session456"

        # After context
        assert _log_chat_id.get() is None
        assert _log_session_id.get() is None

    def test_log_context_with_only_chat_id(self):
        from ash.logging import _log_chat_id, _log_session_id, log_context

        with log_context(chat_id="chat123"):
            assert _log_chat_id.get() == "chat123"
            assert _log_session_id.get() is None

    def test_log_context_with_only_session_id(self):
        from ash.logging import _log_chat_id, _log_session_id, log_context

        with log_context(session_id="session456"):
            assert _log_chat_id.get() is None
            assert _log_session_id.get() == "session456"

    def test_log_context_nested(self):
        from ash.logging import _log_chat_id, _log_session_id, log_context

        with log_context(chat_id="outer_chat"):
            assert _log_chat_id.get() == "outer_chat"

            with log_context(chat_id="inner_chat", session_id="inner_session"):
                assert _log_chat_id.get() == "inner_chat"
                assert _log_session_id.get() == "inner_session"

            # Back to outer
            assert _log_chat_id.get() == "outer_chat"
            assert _log_session_id.get() is None


class TestComponentFormatter:
    """Tests for ComponentFormatter with context injection."""

    def test_extracts_component_from_ash_logger(self):
        import logging

        from ash.logging import ComponentFormatter

        formatter = ComponentFormatter("%(component)s: %(message)s")
        record = logging.LogRecord(
            name="ash.providers.telegram",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "providers:" in result

    def test_extracts_component_from_non_ash_logger(self):
        import logging

        from ash.logging import ComponentFormatter

        formatter = ComponentFormatter("%(component)s: %(message)s")
        record = logging.LogRecord(
            name="httpx.client",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "httpx:" in result

    def test_injects_context_when_available(self):
        import logging

        from ash.logging import ComponentFormatter, log_context

        formatter = ComponentFormatter("%(context)s%(component)s: %(message)s")
        record = logging.LogRecord(
            name="ash.agents",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        with log_context(chat_id="-542863895"):
            result = formatter.format(record)
            assert "[-5428638]" in result

    def test_no_context_when_not_set(self):
        import logging

        from ash.logging import ComponentFormatter

        formatter = ComponentFormatter("%(context)s%(component)s: %(message)s")
        record = logging.LogRecord(
            name="ash.agents",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        # Should not have brackets when no context
        assert "[" not in result or "[]" in result  # Allow empty brackets

    def test_context_with_session_id_different_from_chat_id(self):
        import logging

        from ash.logging import ComponentFormatter, log_context

        formatter = ComponentFormatter("%(context)s%(component)s: %(message)s")
        record = logging.LogRecord(
            name="ash.agents",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        with log_context(chat_id="-542863895", session_id="abc123def456"):
            result = formatter.format(record)
            # Should have both chat and session
            assert "-5428638" in result
            assert "s:abc123" in result
