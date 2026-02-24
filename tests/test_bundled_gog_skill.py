from pathlib import Path

from ash.config.models import SkillConfig
from ash.skills import SkillRegistry


def _load_gog_skill_text() -> str:
    return Path("src/ash/skills/bundled/gog/SKILL.md").read_text()


def test_gog_skill_is_opt_in_and_hidden_by_default() -> None:
    registry = SkillRegistry()
    registry.discover(
        Path(),
        include_bundled=True,
        include_installed=False,
        include_user=False,
    )

    assert not registry.has("gog")


def test_gog_skill_is_available_when_enabled() -> None:
    registry = SkillRegistry(skill_config={"gog": SkillConfig(enabled=True)})
    registry.discover(
        Path(),
        include_bundled=True,
        include_installed=False,
        include_user=False,
    )

    skill = registry.get("gog")
    assert skill.opt_in is True
    assert skill.sensitive is True
    assert skill.allowed_chat_types == ["private"]
    assert skill.capabilities == ["gog.email", "gog.calendar"]
    assert skill.allowed_tools == ["bash"]


def test_gog_skill_uses_capability_contract_text() -> None:
    text = _load_gog_skill_text()

    assert "ash-sb capability" in text
    assert "[skills.gog.capability_provider]" in text
    assert "Never read or request raw OAuth access tokens" in text
