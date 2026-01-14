# Codebase Comparison: ash vs archer vs clawdbot vs pi-mono

Comprehensive comparison of four related AI agent codebases.

## Codebases

| Project | Language | LOC | Description |
|---------|----------|-----|-------------|
| **ash** | Python | 22K | Personal assistant with SQLite memory, Docker sandbox |
| **archer** | TypeScript | 4.7K | Telegram bot using pi-* libraries |
| **clawdbot** | TypeScript | 209K | Multi-provider platform (7 messaging providers) |
| **pi-mono** | TypeScript | - | Underlying libraries (pi-ai, pi-agent-core, etc.) |

## Comparison Documents

### Core Systems
1. [Core Agent Loop](01-core-agent-loop.md) - Agent orchestration, tool execution, event handling
2. [LLM Integration](02-llm-integration.md) - Provider support, streaming, thinking/reasoning
3. [Tool System](03-tool-system.md) - Available tools, schemas, execution model
4. [Session Management](04-session-management.md) - Persistence, compaction, history

### Data & Security
5. [Memory System](05-memory-system.md) - Storage, retrieval, semantic search
6. [Sandbox Execution](06-sandbox-execution.md) - Docker vs host, security model
7. [Provider Integrations](07-provider-integrations.md) - Telegram, Slack, Discord, etc.
8. [Skills System](08-skills-system.md) - User-defined behaviors, discovery

### Infrastructure
9. [Events & Scheduling](09-events-scheduling.md) - Cron, one-shot, immediate events
10. [Configuration](10-configuration.md) - TOML/JSON, validation, hot reload
11. [CLI Interface](11-cli-interface.md) - Commands, frameworks, structure
12. [Logging & Observability](12-logging-observability.md) - Structured logging, error reporting

## Key Takeaways

### ash Strengths
- **Memory system**: Only codebase with semantic search via sqlite-vec
- **Sandbox security**: Most hardened Docker sandbox (read-only rootfs, caps dropped, gVisor)
- **Person tracking**: Relationship extraction and person entity management
- **Sentry integration**: Production error tracking

### archer/pi-mono Strengths
- **Parallel tool execution**: Concurrent tool calls for better latency
- **Rich event system**: 14+ event types for fine-grained UI updates
- **Session branching**: Tree-based sessions with fork/branch support
- **Unified LLM abstraction**: pi-ai supports 7+ providers with consistent API

### clawdbot Strengths
- **Provider breadth**: 7 messaging providers (Telegram, Slack, Discord, WhatsApp, Signal, iMessage, Teams)
- **Model failover**: Cascade fallback chains for resilience
- **51 bundled skills**: Rich ecosystem out of the box
- **Hot config reload**: Update config without restart
- **Secret redaction**: Pattern-based log sanitization

## Repository Paths

- **ash**: `/home/dcramer/src/ash`
- **archer**: `/home/dcramer/src/archer`
- **clawdbot**: `/home/dcramer/src/clawdbot`
- **pi-mono**: `/home/dcramer/src/pi-mono`
