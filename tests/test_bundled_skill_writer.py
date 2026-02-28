from pathlib import Path

from ash.skills import SkillRegistry


def _load_skill_writer_text() -> str:
    skill_path = Path("src/ash/skills/bundled/skill-writer/SKILL.md")
    return skill_path.read_text()


def test_skill_writer_max_iterations_is_60() -> None:
    registry = SkillRegistry()
    registry.discover(Path(), include_bundled=True)
    skill = registry.get("skill-writer")
    assert skill.max_iterations == 60


def test_skill_writer_has_ask_only_if_non_obvious_policy() -> None:
    text = _load_skill_writer_text()
    assert "Clarification Policy (Ask Only If Non-Obvious)" in text
    assert "Ask only when ambiguity materially affects correctness" in text
    assert "If the user does not answer, proceed with explicit assumptions" in text


def test_skill_writer_uses_references_conditionally() -> None:
    text = _load_skill_writer_text()
    assert "Load references only as needed" in text
    assert "references/skills-spec.md" in text
    assert "references/example-skill.md" in text


def test_skill_writer_quality_gates_present() -> None:
    text = _load_skill_writer_text()
    assert "## Quality Gates" in text
    assert "Description quality: include concrete trigger contexts" in text
    assert "Progressive disclosure: keep core workflow in SKILL.md" in text
    assert "Tool minimalism: include only required tools in `allowed_tools`." in text
