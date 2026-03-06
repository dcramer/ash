from pathlib import Path

from ash.config.models import SkillConfig
from ash.skills import SkillRegistry


def _load_google_skill_text() -> str:
    return Path("src/ash/integrations/skills/capabilities/google/SKILL.md").read_text()


def test_google_skill_is_opt_in_and_hidden_by_default() -> None:
    registry = SkillRegistry()
    registry.discover(
        Path(),
        include_bundled=True,
        include_installed=False,
        include_user=False,
    )

    assert not registry.has("google")


def test_google_skill_is_available_when_enabled() -> None:
    registry = SkillRegistry(skill_config={"google": SkillConfig(enabled=True)})
    registry.discover(
        Path(),
        include_bundled=True,
        include_installed=False,
        include_user=False,
    )

    skill = registry.get("google")
    assert skill.opt_in is True
    assert skill.sensitive is True
    assert skill.allowed_chat_types == ["private"]
    assert skill.capabilities == ["gog.email", "gog.calendar"]
    assert skill.allowed_tools == ["bash"]


def test_google_skill_uses_capability_contract_text() -> None:
    text = _load_google_skill_text()

    assert "ash-sb capability" in text
    assert "[skills.google]" in text
    assert "Never read or request raw OAuth access tokens" in text


def test_google_skill_includes_summary_and_day_at_a_glance_playbooks() -> None:
    text = _load_google_skill_text().lower()
    assert "summarize emails" in text
    assert "day at a glance" in text
    assert "get_message" in text


def test_google_skill_includes_archive_and_label_mutation_guidance() -> None:
    text = _load_google_skill_text().lower()
    assert "archive_messages" in text
    assert "update_labels" in text
    assert "always confirm key details" in text


def test_google_skill_avoids_auth_loop_and_includes_flow_recovery_guidance() -> None:
    text = _load_google_skill_text().lower()
    assert "flow id: <flow_id>" in text
    assert (
        "do not start a new `auth begin` while a valid callback/code is present" in text
    )
    assert "ash-sb capability auth list" in text
