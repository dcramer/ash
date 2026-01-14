# LLM Integration Comparison

This document compares LLM integration patterns across four codebases: ash, archer, clawdbot, and pi-mono.

## Overview

LLM integration encompasses:
- **Provider abstraction**: How different LLM APIs (Anthropic, OpenAI, Google, etc.) are unified
- **Rate limiting**: Preventing API overload and managing concurrent requests
- **Streaming**: Real-time token delivery for responsive UX
- **Extended thinking**: Reasoning/thinking budget configuration
- **Failover**: Graceful degradation when providers fail
- **Authentication**: API keys, OAuth tokens, credential management

## Comparison Table

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Language** | Python | TypeScript | TypeScript | TypeScript |
| **Providers** | Anthropic, OpenAI | Anthropic (via pi-ai) | Anthropic, OpenAI, Google, Vertex, OpenRouter, XAI, Mistral | 7+ (Anthropic, OpenAI, Google, Vertex, Groq, XAI, etc.) |
| **Provider Abstraction** | ABC base class | Uses pi-ai library | Uses pi-ai library | Unified `stream()` API |
| **Rate Limiting** | Semaphore (2 concurrent) | None (via pi-ai) | Profile rotation + cooldowns | None |
| **Streaming** | AsyncGenerator + StreamChunk | Via pi-ai | Via pi-ai | Event stream with typed events |
| **Thinking/Reasoning** | ThinkingConfig with budget levels | `thinkingLevel: "off"` | Via pi-ai SimpleStreamOptions | ThinkingLevel: off/minimal/low/medium/high/xhigh |
| **Failover** | Retry with backoff | Via pi-agent-core retry | `runWithModelFallback()` cascade | None built-in |
| **Auth** | API key from env | AuthStorage + OAuth | Multi-profile OAuth/API key | `getEnvApiKey()` per provider |

## Detailed Analysis

### 1. ash (Python)

**Architecture**: Clean ABC-based provider abstraction with separate implementations for each provider.

**Provider Base Class** (`src/ash/llm/base.py`):
```python
class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'anthropic', 'openai')."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: "ThinkingConfig | None" = None,
    ) -> CompletionResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: "ThinkingConfig | None" = None,
    ) -> AsyncGenerator[StreamChunk, None]: ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]: ...
```

**Rate Limiting** (`src/ash/llm/anthropic.py`):
```python
class AnthropicProvider(LLMProvider):
    _semaphore: asyncio.Semaphore | None = None
    _max_concurrent: int = 2  # Max concurrent API requests

    async def complete(self, ...):
        assert self._semaphore is not None
        async def _make_request() -> anthropic.types.Message:
            async with self._semaphore:
                logger.debug(f"Acquired API slot, calling {model_name}")
                response = await self._client.messages.create(**kwargs)
                return response

        response = await with_retry(
            _make_request,
            config=RetryConfig(enabled=True, max_retries=3),
            operation_name=f"Anthropic {model_name}",
        )
```

**Extended Thinking** (`src/ash/llm/thinking.py`):
```python
class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"   # 1K tokens
    LOW = "low"           # 4K tokens
    MEDIUM = "medium"     # 16K tokens
    HIGH = "high"         # 64K tokens

@dataclass
class ThinkingConfig:
    level: ThinkingLevel = ThinkingLevel.OFF
    budget_tokens: int | None = None

    def to_api_params(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.effective_budget,
            }
        }
```

**Retry Logic** (`src/ash/llm/retry.py`):
```python
RETRYABLE_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|429|500|502|503|504|..."
)

async def with_retry[T](
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    operation_name: str = "API call",
) -> T:
    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if not is_retryable_error(e):
                raise
            delay_ms = min(config.base_delay_ms * (2**attempt), config.max_delay_ms)
            await asyncio.sleep(delay_ms / 1000)
```

**Streaming Types** (`src/ash/llm/types.py`):
```python
class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"
```

---

### 2. archer (TypeScript)

**Architecture**: Thin wrapper using pi-ai library for LLM calls. Focused on Telegram bot integration.

**LLM Integration** (`src/agent.ts`):
```typescript
import { completeSimple, getModel, type ImageContent } from "@mariozechner/pi-ai";

// Hardcoded models
const model = getModel("anthropic", "claude-sonnet-4-5");
const haikuModel = getModel("anthropic", "claude-haiku-4-5");

// Agent creation with pi-agent-core
const agent = new Agent({
    initialState: {
        systemPrompt,
        model,
        thinkingLevel: "off",
        tools,
    },
    convertToLlm,
    getApiKey: async () => getAnthropicApiKey(authStorage),
});
```

**Authentication**:
```typescript
async function getAnthropicApiKey(authStorage: AuthStorage): Promise<string> {
    const key = await authStorage.getApiKey("anthropic");
    if (!key) {
        throw new Error(
            "No API key found for anthropic.\n\n" +
            "Set an API key environment variable, or use /login with Anthropic..."
        );
    }
    return key;
}
```

**Characteristics**:
- Uses pi-ai's `completeSimple()` for non-streaming and pi-agent-core's Agent class for agentic loops
- OAuth support via AuthStorage
- Delegates all LLM complexity to external libraries
- Simple model selection (hardcoded to Claude Sonnet 4.5)

---

### 3. clawdbot (TypeScript)

**Architecture**: Sophisticated multi-provider system with auth profiles, failover, and usage tracking.

**Auth Profiles** (`src/agents/auth-profiles.ts`):
```typescript
export type ApiKeyCredential = {
    type: "api_key";
    provider: string;
    key: string;
    email?: string;
};

export type TokenCredential = {
    type: "token";
    provider: string;
    token: string;
    expires?: number;
};

export type OAuthCredential = OAuthCredentials & {
    type: "oauth";
    provider: OAuthProvider;
    email?: string;
};

export type ProfileUsageStats = {
    lastUsed?: number;
    cooldownUntil?: number;
    disabledUntil?: number;
    disabledReason?: AuthProfileFailureReason;
    errorCount?: number;
};
```

**Round-Robin Profile Selection**:
```typescript
function orderProfilesByMode(order: string[], store: AuthProfileStore): string[] {
    const available: string[] = [];
    const inCooldown: string[] = [];

    for (const profileId of order) {
        if (isProfileInCooldown(store, profileId)) {
            inCooldown.push(profileId);
        } else {
            available.push(profileId);
        }
    }

    // Sort by type preference (oauth > token > api_key), then by lastUsed (round-robin)
    const scored = available.map((profileId) => {
        const type = store.profiles[profileId]?.type;
        const typeScore = type === "oauth" ? 0 : type === "token" ? 1 : 2;
        const lastUsed = store.usageStats?.[profileId]?.lastUsed ?? 0;
        return { profileId, typeScore, lastUsed };
    });

    return [...sorted, ...cooldownSorted];
}
```

**Model Failover** (`src/agents/model-fallback.ts`):
```typescript
export async function runWithModelFallback<T>(params: {
    cfg: ClawdbotConfig | undefined;
    provider: string;
    model: string;
    run: (provider: string, model: string) => Promise<T>;
    onError?: (attempt: {...}) => void | Promise<void>;
}): Promise<{result: T; provider: string; model: string; attempts: FallbackAttempt[]}> {
    const candidates = resolveFallbackCandidates(params);
    const attempts: FallbackAttempt[] = [];

    for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        try {
            const result = await params.run(candidate.provider, candidate.model);
            return { result, provider: candidate.provider, model: candidate.model, attempts };
        } catch (err) {
            if (isAbortError(err)) throw err;
            if (!isFailoverError(normalized)) throw err;
            attempts.push({
                provider: candidate.provider,
                model: candidate.model,
                error: described.message,
                reason: described.reason,
            });
        }
    }
    throw new Error(`All models failed (${attempts.length}): ${summary}`);
}
```

**Cooldown Calculation**:
```typescript
export function calculateAuthProfileCooldownMs(errorCount: number): number {
    const normalized = Math.max(1, errorCount);
    return Math.min(
        60 * 60 * 1000, // 1 hour max
        60 * 1000 * 5 ** Math.min(normalized - 1, 3),  // 1min, 5min, 25min
    );
}
```

**Characteristics**:
- Enterprise-grade auth with OAuth, API keys, and tokens
- Automatic credential sync with Claude CLI and Codex CLI
- Round-robin profile rotation with cooldown tracking
- Model failover cascade across multiple providers
- Billing failure detection with extended backoff

---

### 4. pi-mono (TypeScript)

**Architecture**: Unified multi-provider API with typed streaming events.

**Stream Entry Point** (`packages/ai/src/stream.ts`):
```typescript
export function stream<TApi extends Api>(
    model: Model<TApi>,
    context: Context,
    options?: OptionsForApi<TApi>,
): AssistantMessageEventStream {
    const apiKey = options?.apiKey || getEnvApiKey(model.provider);

    switch (model.api) {
        case "anthropic-messages":
            return streamAnthropic(model, context, providerOptions);
        case "openai-completions":
            return streamOpenAICompletions(model, context, providerOptions);
        case "openai-responses":
            return streamOpenAIResponses(model, context, providerOptions);
        case "google-generative-ai":
            return streamGoogle(model, context, providerOptions);
        case "google-vertex":
            return streamGoogleVertex(model, context, providerOptions);
        // ...
    }
}
```

**Thinking Levels** (`packages/ai/src/types.ts`):
```typescript
export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

export interface ThinkingBudgets {
    minimal?: number;
    low?: number;
    medium?: number;
    high?: number;
}
```

**Provider-Specific Thinking Mapping**:
```typescript
// Anthropic
if (options?.thinkingEnabled && model.reasoning) {
    params.thinking = {
        type: "enabled",
        budget_tokens: options.thinkingBudgetTokens || 1024,
    };
}

// Google Gemini 3
if (isGemini3ProModel(model)) {
    return {
        thinking: {
            enabled: true,
            level: getGemini3ThinkingLevel(effort, model),  // "LOW" | "HIGH"
        },
    };
}

// OpenAI
params.reasoningEffort = options?.reasoning;  // minimal/low/medium/high
```

**Streaming Event Types**:
```typescript
export type AssistantMessageEvent =
    | { type: "start"; partial: AssistantMessage }
    | { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
    | { type: "done"; reason: StopReason; message: AssistantMessage }
    | { type: "error"; reason: StopReason; error: AssistantMessage };
```

**OAuth Stealth Mode** (`packages/ai/src/providers/anthropic.ts`):
```typescript
// When using OAuth tokens, mimic Claude Code's headers
if (isOAuthToken(apiKey)) {
    const defaultHeaders = {
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        "user-agent": `claude-cli/${claudeCodeVersion} (external, cli)`,
        "x-app": "cli",
    };
    // Tool names mapped to Claude Code equivalents
    const claudeCodeToolNames = {
        read: "Read", write: "Write", edit: "Edit", bash: "Bash"
    };
}
```

**Characteristics**:
- Provider-agnostic streaming with unified event types
- Built-in support for 7+ providers
- Sophisticated thinking level mapping per provider
- OAuth token detection and Claude Code compatibility mode
- Exhaustive TypeScript types for compile-time API coverage checks

---

## Key Differences

### 1. Provider Abstraction

| Approach | Codebase | Trade-offs |
|----------|----------|------------|
| ABC base class | ash | Clean interfaces, explicit contracts, Python-idiomatic |
| External library | archer | Minimal code, depends on library updates |
| Unified switch | pi-mono | Single entry point, explicit provider mapping |

### 2. Rate Limiting

| Approach | Codebase | Trade-offs |
|----------|----------|------------|
| Semaphore | ash | Simple, process-local, fixed concurrency |
| Profile rotation | clawdbot | Distributed, handles billing/rate limits differently |
| None | pi-mono, archer | Relies on provider-side limits |

### 3. Failover Strategy

| Approach | Codebase | Trade-offs |
|----------|----------|------------|
| Retry only | ash | Simple, same model retried |
| Model cascade | clawdbot | Resilient, can switch providers |
| None | pi-mono | Caller handles failures |

### 4. Thinking Configuration

| Approach | Codebase | Trade-offs |
|----------|----------|------------|
| Enum + dataclass | ash | Type-safe, self-documenting |
| String levels | archer, clawdbot | Simple, matches API naming |
| Per-provider mapping | pi-mono | Handles provider differences (budgets vs levels) |

### 5. Authentication

| Approach | Codebase | Trade-offs |
|----------|----------|------------|
| Env vars | ash, pi-mono | Simple, standard |
| OAuth profiles | clawdbot | Enterprise-ready, token refresh |
| AuthStorage | archer | Persistent, supports OAuth |

---

## Recommendations for ash

Based on this analysis, ash could consider adopting:

### 1. Model Failover (from clawdbot)

Add a model fallback cascade when the primary model fails:

```python
@dataclass
class FallbackConfig:
    candidates: list[tuple[str, str]]  # (provider, model) pairs
    retryable_errors: set[str] = field(default_factory=lambda: {"overloaded", "rate_limit"})

async def with_model_fallback(
    providers: dict[str, LLMProvider],
    fallback: FallbackConfig,
    request: Callable[[LLMProvider, str], Awaitable[T]],
) -> tuple[T, str, str]:
    """Try models in order until one succeeds."""
    attempts = []
    for provider_name, model in fallback.candidates:
        try:
            return await request(providers[provider_name], model), provider_name, model
        except Exception as e:
            if not is_failover_error(e):
                raise
            attempts.append((provider_name, model, str(e)))
    raise AllModelsFailed(attempts)
```

### 2. xhigh Thinking Level (from pi-mono)

Add an "xhigh" thinking level for o1/o3-style reasoning models:

```python
class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"   # 1K tokens
    LOW = "low"           # 4K tokens
    MEDIUM = "medium"     # 16K tokens
    HIGH = "high"         # 64K tokens
    XHIGH = "xhigh"       # 128K+ tokens (for reasoning models)
```

### 3. Unified Streaming Events (from pi-mono)

The current ash streaming is good but could add `thinking_delta` events:

```python
class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"  # Add this
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"
```

### 4. Cooldown Tracking (from clawdbot)

Track per-provider failure state to implement smarter backoff:

```python
@dataclass
class ProviderState:
    last_used: datetime | None = None
    cooldown_until: datetime | None = None
    error_count: int = 0

    @property
    def in_cooldown(self) -> bool:
        if not self.cooldown_until:
            return False
        return datetime.now() < self.cooldown_until

    def mark_failure(self) -> None:
        self.error_count += 1
        backoff = min(60 * 60, 60 * (5 ** min(self.error_count - 1, 3)))
        self.cooldown_until = datetime.now() + timedelta(seconds=backoff)
```

### 5. NOT Recommended

- **OAuth profiles**: Adds complexity ash doesn't need (no enterprise use case)
- **External library dependency**: ash's explicit provider code is clearer
- **Claude Code stealth mode**: Only needed for OAuth token usage

---

## Summary

Each codebase optimizes for different use cases:

- **ash**: Clean, minimal, Python-idiomatic. Good for single-user personal assistant.
- **archer**: Thin wrapper for Telegram bot. Delegates complexity to libraries.
- **clawdbot**: Enterprise-grade with auth profiles, failover, usage tracking.
- **pi-mono**: Library for other apps. Maximum provider coverage, typed events.

The most valuable patterns ash could adopt are **model failover** and **cooldown tracking** from clawdbot, and **thinking_delta events** from pi-mono. These add resilience without significant complexity.
