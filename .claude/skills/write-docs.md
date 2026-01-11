# /write-docs

Create or update Ash documentation pages following project conventions.

## Usage

```
/write-docs <page>
```

Where `<page>` is a path like `cli/chat` or `configuration/models`.

## Process

1. Read existing page if present: `docs/src/content/docs/<page>.mdx`
2. Read source files referenced in content:
   - CLI: `src/ash/cli/app.py`
   - Config: `src/ash/config/models.py`
   - Architecture: `ARCHITECTURE.md` and relevant `src/ash/` modules
3. Read related specs from `specs/` directory
4. Draft documentation following patterns below
5. Verify accuracy against source code
6. Write the page to `docs/src/content/docs/<page>.mdx`
7. Run `cd docs && pnpm build` to verify

## Frontmatter

Every page requires:

```yaml
---
title: Page Title
description: One-line description for SEO
sidebar:
  order: 1  # Optional, controls position within section
---
```

## Style Guidelines

### DO
- Use active voice: "Run the command" not "The command can be run"
- Start sections with actions: "Install", "Configure", "Run"
- Include working examples from actual code
- Reference source files when documenting behavior
- Use Starlight components for structure
- Keep paragraphs short - users skim

### DO NOT
- Include design rationale (that belongs in specs)
- Use marketing language or superlatives
- Assume reader knowledge - define terms on first use
- Document hypothetical or planned features

## Code Examples

Always use fenced blocks with language tags:

```bash
ash chat "Hello, world"
```

```toml
[models.default]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
```

```python
from ash.tools.base import Tool, ToolResult

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"
```

## Starlight Components

Import at top of MDX file when needed:

```mdx
import { Tabs, TabItem, Card, Aside } from '@astrojs/starlight/components';
```

Aside for tips/warnings:

```mdx
<Aside type="tip">
  Use `--no-streaming` for CI/CD pipelines.
</Aside>

<Aside type="caution">
  This will delete all data.
</Aside>
```

Tabs for alternatives:

```mdx
<Tabs>
  <TabItem label="uv">
    ```bash
    uv tool install ash-agent
    ```
  </TabItem>
  <TabItem label="pip">
    ```bash
    pip install ash-agent
    ```
  </TabItem>
</Tabs>
```

## Tables

Use tables for reference data:

```mdx
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `default` | Model alias to use |
| `--config` | `~/.ash/config.toml` | Config file path |
```

## Verification

After writing, verify the build:

```bash
cd docs && pnpm build
```

Check for:
- Build errors
- Broken links
- Missing imports
- Correct sidebar ordering
