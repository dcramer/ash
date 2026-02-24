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

## Testing Checklist

1. Add unit tests for hook behavior.
2. Add runtime integration tests for ordering and side effects.
3. Add/update architecture guards when ownership boundaries change.
