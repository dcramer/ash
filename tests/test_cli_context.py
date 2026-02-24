"""Tests for CLI context helpers."""

from ash.cli.context import generate_config_template


def test_generate_config_template_includes_default_fast_and_codex_models() -> None:
    template = generate_config_template()

    assert "[models.default]" in template
    assert 'model = "gpt-5.2"' in template

    assert "[models.fast]" in template
    assert "currently unsupported on openai-oauth" in template
    assert 'model = "gpt-5.2"' in template

    assert "[models.codex]" in template
    assert 'model = "gpt-5.2-codex"' in template
