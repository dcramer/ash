# Integrations

Integration contributors extend runtime behavior through deterministic hooks.

## Contributor Template

Use `IntegrationContributor` and implement only hooks your feature needs:

```python
from ash.integrations.runtime import IntegrationContributor, IntegrationContext


class ExampleIntegration(IntegrationContributor):
    name = "example"
    priority = 500

    async def setup(self, context: IntegrationContext) -> None:
        ...

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        ...

    def augment_prompt_context(self, prompt_context, session, context):
        return prompt_context

    def augment_skill_instructions(self, skill_name, context):
        return []

    def augment_sandbox_env(self, env, session, effective_user_id, context):
        return env

    async def preprocess_incoming_message(self, message, context: IntegrationContext):
        return message

    async def on_message_postprocess(
        self,
        user_message: str,
        session,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> None:
        ...
```

## Rules

1. Set stable `name` and `priority`; runtime ordering is `(priority, name)`.
2. Keep hook behavior local to the integration domain.
3. Post-turn behavior belongs in `on_message_postprocess`, not provider/core call sites.
4. Pre-turn inbound transformations belong in `preprocess_incoming_message`.
5. Register via shared composition (`create_default_integrations` + `compose_integrations`).
6. Hook failures must be isolated per contributor and logged with hook + contributor metadata.
7. Contributors that fail in `setup` are excluded from later hook/lifecycle execution.
8. If an integration introduces graph-backed entities, it owns registration of node collections and edge schemas via graph extension APIs before use.
9. Integration contributors are trusted first-party runtime capabilities, not third-party plugin points.
10. Third-party extensions must go through skills/capability surfaces rather than registering integration contributors directly.
11. Runtime-scoped sandbox env data (for example RPC transport hints) must flow through `IntegrationContext.sandbox_env` and integration env hooks, not process-global environment mutation.

## Integration-Provided Skills

Integrations can provide skills from a co-located `skills/` directory at
`src/ash/integrations/skills/{contributor_name}/{skill_name}/SKILL.md`.

These are loaded during `SkillRegistry.discover()` at integration precedence
(above bundled, below installed/user/workspace) and gated on the same
`include_bundled` flag as built-in skills.

Layout:
```
src/ash/integrations/skills/
  todo/           # contributor (matches integration name)
    todo/         # skill name
      SKILL.md
  browser/
    screenshot/
      SKILL.md
```

Each contributor directory is iterated in sorted order. Skill files follow
the standard `SKILL.md` frontmatter format used by all other skill sources.

### Container Mount Paths

Skills are mounted read-only in the sandbox container at predictable paths:

```
/ash/skills/{skill_name}/                                    # bundled (ro)
/ash/integrations/{contributor}/skills/{skill_name}/          # integration (ro)
/workspace/skills/{skill_name}/                              # workspace (rw)
```

The system prompt includes the sandbox path for each skill so the agent can
locate co-located files. The `ASH_SKILL_DIRS` env var (colon-separated) lists
all mounted skill directories for `ash-sb skill list` discovery.

## Testing Checklist

1. Add unit tests for hook behavior.
2. Add runtime integration tests for ordering and side effects.
3. Add/update architecture guards when ownership boundaries change.
