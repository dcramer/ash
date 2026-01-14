# LLM Integration Gap Analysis

This document analyzes 6 specific gaps between ash's LLM provider implementation and the reference implementations in clawdbot and pi-mono.

## Files Analyzed

**Ash:**
- `/home/dcramer/src/ash/src/ash/llm/base.py` - Abstract LLM provider interface
- `/home/dcramer/src/ash/src/ash/llm/anthropic.py` - Anthropic Claude provider
- `/home/dcramer/src/ash/src/ash/llm/openai.py` - OpenAI provider
- `/home/dcramer/src/ash/src/ash/llm/thinking.py` - Extended thinking configuration
- `/home/dcramer/src/ash/src/ash/llm/retry.py` - Retry utilities
- `/home/dcramer/src/ash/src/ash/llm/types.py` - Message and stream types

**References:**
- `/home/dcramer/src/pi-mono/packages/ai/src/stream.ts` - Unified streaming with thinking level abstraction
- `/home/dcramer/src/pi-mono/packages/ai/src/types.ts` - Stream event types including thinking events
- `/home/dcramer/src/clawdbot/src/agents/model-fallback.ts` - Model failover cascade
- `/home/dcramer/src/clawdbot/src/agents/auth-profiles.ts` - Auth profile rotation with cooldown tracking
- `/home/dcramer/src/clawdbot/src/agents/failover-error.ts` - Failover error classification
- `/home/dcramer/src/clawdbot/src/agents/pi-embedded-helpers.ts` - Error pattern matching

---

## Gap 1: Model Failover Cascade

### What Ash is Missing

Ash's retry mechanism only retries the **same** provider and model on failure. It has no concept of falling back to alternative providers or models when one fails.

Current ash code (`retry.py` lines 68-126):
```python
async def with_retry[T](
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    operation_name: str = "API call",
) -> T:
    # Just retries the same operation, no fallback to different models
    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if not is_retryable_error(e):
                raise
            # Wait and retry same operation
            await asyncio.sleep(delay_s)
```

And in `anthropic.py` (lines 230-235):
```python
response = await with_retry(
    _make_request,
    config=RetryConfig(enabled=True, max_retries=3),
    operation_name=f"Anthropic {model_name}",
)
```

Clawdbot has `runWithModelFallback()` that tries a cascade of provider/model combinations:
1. Primary model from config
2. Fallback models from `model.fallbacks` array
3. Falls back to default model as last resort

### Reference

**Best implementation:** clawdbot (`model-fallback.ts` lines 187-264)
```typescript
export async function runWithModelFallback<T>(params: {
  cfg: ClawdbotConfig | undefined;
  provider: string;
  model: string;
  run: (provider: string, model: string) => Promise<T>;
  onError?: (attempt: {...}) => void | Promise<void>;
}): Promise<{
  result: T;
  provider: string;
  model: string;
  attempts: FallbackAttempt[];
}> {
  const candidates = resolveFallbackCandidates(params);
  const attempts: FallbackAttempt[] = [];

  for (let i = 0; i < candidates.length; i += 1) {
    const candidate = candidates[i];
    try {
      const result = await params.run(candidate.provider, candidate.model);
      return { result, provider: candidate.provider, model: candidate.model, attempts };
    } catch (err) {
      if (isAbortError(err)) throw err;
      const normalized = coerceToFailoverError(err, {...}) ?? err;
      if (!isFailoverError(normalized)) throw err;

      attempts.push({...});
      await params.onError?.({...});
    }
  }
  throw new Error(`All models failed (${attempts.length}): ${summary}`);
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/failover.py` (new file)
- `/home/dcramer/src/ash/src/ash/llm/types.py`
- `/home/dcramer/src/ash/src/ash/config/models.py`

### Proposed Changes

```python
# New file: src/ash/llm/failover.py
"""Model failover cascade for LLM requests."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from ash.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class FailoverReason(str, Enum):
    """Reason for failover to next model."""
    AUTH = "auth"
    FORMAT = "format"
    RATE_LIMIT = "rate_limit"
    BILLING = "billing"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class FailoverError(Exception):
    """Error that should trigger failover to next model."""
    message: str
    reason: FailoverReason
    provider: str | None = None
    model: str | None = None
    status: int | None = None
    code: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass
class ModelCandidate:
    """A provider/model combination to try."""
    provider: str
    model: str


@dataclass
class FallbackAttempt:
    """Record of a failed model attempt."""
    provider: str
    model: str
    error: str
    reason: FailoverReason | None = None
    status: int | None = None


@dataclass
class FallbackResult[T]:
    """Result of failover cascade."""
    result: T
    provider: str
    model: str
    attempts: list[FallbackAttempt] = field(default_factory=list)


def classify_failover_reason(error: Exception) -> FailoverReason | None:
    """Classify an error to determine if failover should occur.

    Returns None if the error should not trigger failover.
    """
    error_str = str(error).lower()

    # Rate limit patterns
    rate_limit_patterns = [
        "rate_limit", "rate limit", "too many requests", "429",
        "exceeded your current quota", "resource has been exhausted",
        "quota exceeded", "resource_exhausted", "usage limit",
    ]
    if any(p in error_str for p in rate_limit_patterns):
        return FailoverReason.RATE_LIMIT

    # Billing patterns
    billing_patterns = [
        "402", "payment required", "insufficient credits",
        "credit balance", "plans & billing",
    ]
    if any(p in error_str for p in billing_patterns):
        return FailoverReason.BILLING

    # Auth patterns
    auth_patterns = [
        "invalid_api_key", "invalid api key", "incorrect api key",
        "invalid token", "authentication", "unauthorized",
        "forbidden", "access denied", "401", "403",
    ]
    if any(p in error_str for p in auth_patterns):
        return FailoverReason.AUTH

    # Timeout patterns
    timeout_patterns = [
        "timeout", "timed out", "deadline exceeded",
    ]
    if any(p in error_str for p in timeout_patterns):
        return FailoverReason.TIMEOUT

    # Format/request patterns (model-specific issues)
    format_patterns = [
        "invalid_request_error", "string should match pattern",
        "tool_use.id", "invalid request format",
    ]
    if any(p in error_str for p in format_patterns):
        return FailoverReason.FORMAT

    return None


def is_failover_error(error: Exception) -> bool:
    """Check if an error should trigger failover."""
    return classify_failover_reason(error) is not None


def coerce_to_failover_error(
    error: Exception,
    provider: str | None = None,
    model: str | None = None,
) -> FailoverError | None:
    """Convert an exception to a FailoverError if applicable."""
    if isinstance(error, FailoverError):
        return error

    reason = classify_failover_reason(error)
    if reason is None:
        return None

    # Extract status code if available
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    code = getattr(error, "code", None)

    return FailoverError(
        message=str(error),
        reason=reason,
        provider=provider,
        model=model,
        status=status,
        code=code,
    )


async def run_with_model_fallback[T](
    candidates: list[ModelCandidate],
    run: Callable[[str, str], Awaitable[T]],
    *,
    on_error: Callable[[FallbackAttempt, int, int], Awaitable[None]] | None = None,
) -> FallbackResult[T]:
    """Execute an operation with model fallback on failure.

    Args:
        candidates: List of provider/model combinations to try in order.
        run: Async function that takes (provider, model) and returns result.
        on_error: Optional callback called on each failure.

    Returns:
        FallbackResult with the successful result and attempt history.

    Raises:
        Exception: If all candidates fail, raises the last error or aggregate.
    """
    if not candidates:
        raise ValueError("No model candidates provided")

    attempts: list[FallbackAttempt] = []
    last_error: Exception | None = None

    for i, candidate in enumerate(candidates):
        try:
            result = await run(candidate.provider, candidate.model)
            return FallbackResult(
                result=result,
                provider=candidate.provider,
                model=candidate.model,
                attempts=attempts,
            )
        except Exception as e:
            # Check if this is an abort/cancellation
            if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                raise

            last_error = e
            failover_error = coerce_to_failover_error(e, candidate.provider, candidate.model)

            # If not a failover error, don't try other candidates
            if failover_error is None:
                raise

            attempt = FallbackAttempt(
                provider=candidate.provider,
                model=candidate.model,
                error=str(e),
                reason=failover_error.reason,
                status=failover_error.status,
            )
            attempts.append(attempt)

            logger.warning(
                f"Model {candidate.provider}/{candidate.model} failed "
                f"(attempt {i + 1}/{len(candidates)}): {e}"
            )

            if on_error:
                await on_error(attempt, i + 1, len(candidates))

    # All candidates failed
    if len(attempts) <= 1 and last_error:
        raise last_error

    summary = " | ".join(
        f"{a.provider}/{a.model}: {a.error}" + (f" ({a.reason.value})" if a.reason else "")
        for a in attempts
    )
    raise RuntimeError(f"All models failed ({len(attempts)}): {summary}")


import asyncio  # noqa: E402 - needed for CancelledError check
```

```python
# In config/models.py, add model fallback configuration:

class ModelConfig(BaseModel):
    """Configuration for a model with fallbacks."""

    provider: str
    model: str
    fallbacks: list[str] = Field(default_factory=list)  # "provider/model" strings


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    # Default model configuration
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"

    # Fallback models (tried in order if default fails)
    # Format: ["anthropic/claude-3-5-haiku-20241022", "openai/gpt-4o"]
    fallbacks: list[str] = Field(default_factory=list)
```

### Effort

**M** (half day) - New module with error classification logic, plus config integration.

### Priority

**High** - Critical for reliability. When Anthropic has an outage, ash should fall back to OpenAI automatically.

---

## Gap 2: Unified Thinking Level Abstraction

### What Ash is Missing

Ash has `ThinkingConfig` with `ThinkingLevel` enum (off/minimal/low/medium/high), but:
1. It only applies to Anthropic - OpenAI's `reasoning_effort` is not supported
2. No provider-specific budget mapping (different providers have different valid ranges)
3. No "xhigh" level for providers that support it
4. Thinking config must be passed explicitly - no automatic mapping from a unified "reasoning" parameter

Current ash code (`thinking.py` lines 20-37):
```python
class ThinkingLevel(str, Enum):
    """Thinking budget levels."""
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

THINKING_BUDGETS = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 4096,
    ThinkingLevel.MEDIUM: 16384,
    ThinkingLevel.HIGH: 65536,
}
```

Pi-mono has a unified abstraction that maps across providers:
- Anthropic: Maps to `thinking.budget_tokens`
- OpenAI: Maps to `reasoning_effort`
- Google: Maps to `thinking.budgetTokens` or `thinking.level`

### Reference

**Best implementation:** pi-mono (`stream.ts` lines 179-327, `types.ts` lines 60-68)
```typescript
export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

export interface ThinkingBudgets {
  minimal?: number;
  low?: number;
  medium?: number;
  high?: number;
}

function mapOptionsForApi<TApi extends Api>(
  model: Model<TApi>,
  options?: SimpleStreamOptions,
  apiKey?: string,
): OptionsForApi<TApi> {
  // Helper to clamp xhigh to high for providers that don't support it
  const clampReasoning = (effort: ThinkingLevel | undefined) =>
    (effort === "xhigh" ? "high" : effort);

  switch (model.api) {
    case "anthropic-messages": {
      if (!options?.reasoning) {
        return { ...base, thinkingEnabled: false };
      }
      const defaultBudgets: ThinkingBudgets = {
        minimal: 1024, low: 2048, medium: 8192, high: 16384,
      };
      const budgets = { ...defaultBudgets, ...options?.thinkingBudgets };
      const level = clampReasoning(options.reasoning)!;
      let thinkingBudget = budgets[level]!;
      return {
        ...base,
        maxTokens: Math.min((base.maxTokens || 0) + thinkingBudget, model.maxTokens),
        thinkingEnabled: true,
        thinkingBudgetTokens: thinkingBudget,
      };
    }

    case "openai-completions":
      return {
        ...base,
        reasoningEffort: supportsXhigh(model) ? options?.reasoning : clampReasoning(options?.reasoning),
      };
    // ... similar for google
  }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/thinking.py`
- `/home/dcramer/src/ash/src/ash/llm/openai.py`
- `/home/dcramer/src/ash/src/ash/llm/base.py`

### Proposed Changes

```python
# Modify thinking.py:

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ThinkingLevel(str, Enum):
    """Unified thinking/reasoning level across providers."""
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # For providers that support extra-high reasoning


# Default token budgets per level for token-based providers
DEFAULT_THINKING_BUDGETS = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 2048,
    ThinkingLevel.MEDIUM: 8192,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.XHIGH: 65536,
}

# Provider-specific budget overrides (some providers have different ranges)
PROVIDER_BUDGETS = {
    "anthropic": {
        ThinkingLevel.MINIMAL: 1024,
        ThinkingLevel.LOW: 4096,
        ThinkingLevel.MEDIUM: 16384,
        ThinkingLevel.HIGH: 65536,
        ThinkingLevel.XHIGH: 65536,  # Anthropic doesn't have xhigh, cap at high
    },
    "google-2.5-pro": {
        ThinkingLevel.MINIMAL: 128,
        ThinkingLevel.LOW: 2048,
        ThinkingLevel.MEDIUM: 8192,
        ThinkingLevel.HIGH: 32768,
    },
    "google-2.5-flash": {
        ThinkingLevel.MINIMAL: 128,
        ThinkingLevel.LOW: 2048,
        ThinkingLevel.MEDIUM: 8192,
        ThinkingLevel.HIGH: 24576,
    },
}


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking/reasoning."""

    level: ThinkingLevel = ThinkingLevel.OFF
    budget_tokens: int | None = None  # Override budget for level
    custom_budgets: dict[ThinkingLevel, int] | None = None  # Per-level overrides

    @property
    def enabled(self) -> bool:
        """Check if thinking is enabled."""
        return self.level != ThinkingLevel.OFF

    def get_budget_for_provider(self, provider: str, model: str = "") -> int:
        """Get the token budget for a specific provider.

        Args:
            provider: Provider name (e.g., "anthropic", "openai").
            model: Model name for provider-specific overrides.

        Returns:
            Token budget for the current level.
        """
        if not self.enabled:
            return 0

        if self.budget_tokens is not None:
            return self.budget_tokens

        if self.custom_budgets and self.level in self.custom_budgets:
            return self.custom_budgets[self.level]

        # Check provider-specific budgets
        provider_key = provider.lower()
        if "2.5-pro" in model.lower():
            provider_key = "google-2.5-pro"
        elif "2.5-flash" in model.lower():
            provider_key = "google-2.5-flash"

        if provider_key in PROVIDER_BUDGETS:
            level = self._clamp_level_for_provider(provider_key)
            return PROVIDER_BUDGETS[provider_key].get(level, DEFAULT_THINKING_BUDGETS[level])

        return DEFAULT_THINKING_BUDGETS.get(self.level, 0)

    def _clamp_level_for_provider(self, provider: str) -> ThinkingLevel:
        """Clamp xhigh to high for providers that don't support it."""
        if self.level == ThinkingLevel.XHIGH:
            # Only OpenAI o-series supports xhigh
            if provider not in ("openai",):
                return ThinkingLevel.HIGH
        return self.level

    def to_anthropic_params(self) -> dict[str, Any] | None:
        """Convert to Anthropic API parameters."""
        if not self.enabled:
            return None

        budget = self.get_budget_for_provider("anthropic")
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        }

    def to_openai_params(self) -> dict[str, Any] | None:
        """Convert to OpenAI API parameters (for o-series models)."""
        if not self.enabled:
            return None

        # Map level to OpenAI's reasoning_effort
        level_map = {
            ThinkingLevel.MINIMAL: "low",
            ThinkingLevel.LOW: "low",
            ThinkingLevel.MEDIUM: "medium",
            ThinkingLevel.HIGH: "high",
            ThinkingLevel.XHIGH: "high",  # Will need model check for actual xhigh support
        }

        return {
            "reasoning_effort": level_map.get(self.level, "medium"),
        }

    @classmethod
    def from_level(cls, level: str | ThinkingLevel) -> "ThinkingConfig":
        """Create config from level string or enum."""
        if isinstance(level, str):
            level = ThinkingLevel(level.lower())
        return cls(level=level)


def resolve_thinking(
    param: ThinkingConfig | ThinkingLevel | str | int | None,
) -> ThinkingConfig:
    """Resolve various thinking parameter formats to ThinkingConfig."""
    if param is None:
        return ThinkingConfig()
    if isinstance(param, ThinkingConfig):
        return param
    if isinstance(param, (ThinkingLevel, str)):
        return ThinkingConfig.from_level(param)
    if isinstance(param, int):
        # Find closest level for the budget
        for level in reversed(list(ThinkingLevel)):
            if DEFAULT_THINKING_BUDGETS.get(level, 0) <= param:
                return ThinkingConfig(level=level, budget_tokens=param)
        return ThinkingConfig()
    return ThinkingConfig()
```

```python
# In openai.py, add reasoning support (lines 146-173 build_request_kwargs):

def _build_request_kwargs(
    self,
    messages: list[Message],
    model: str | None,
    tools: list[ToolDefinition] | None,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    thinking: "ThinkingConfig | None" = None,  # Add parameter
    stream: bool = False,
) -> dict[str, Any]:
    """Build common request kwargs for complete and stream methods."""
    model_name = model or self.default_model
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": self._convert_messages(messages, system),
        "max_tokens": max_tokens,
    }

    if stream:
        kwargs["stream"] = True

    if temperature is not None:
        kwargs["temperature"] = temperature

    converted_tools = self._convert_tools(tools)
    if converted_tools:
        kwargs["tools"] = converted_tools

    # Add reasoning support for o-series models
    if thinking and thinking.enabled and self._is_reasoning_model(model_name):
        reasoning_params = thinking.to_openai_params()
        if reasoning_params:
            kwargs.update(reasoning_params)

    return kwargs


def _is_reasoning_model(self, model: str) -> bool:
    """Check if model supports reasoning_effort parameter."""
    model_lower = model.lower()
    return any(p in model_lower for p in ["o1", "o3", "o4"])
```

### Effort

**S** (2-3 hours) - Mostly refactoring existing code with new level and provider-specific mapping.

### Priority

**Medium** - Useful for consistency but current implementation works for Anthropic. Becomes high priority when OpenAI reasoning models are used.

---

## Gap 3: Thinking Delta Events in Stream

### What Ash is Missing

Ash's streaming does not expose thinking content at all. The `StreamEventType` enum has no thinking-related events:

Current ash code (`types.py` lines 24-33):
```python
class StreamEventType(str, Enum):
    """Stream event type."""
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"
```

Pi-mono streams thinking content with dedicated events:
- `thinking_start` - Thinking block begins
- `thinking_delta` - Thinking content chunk
- `thinking_end` - Thinking block complete

This allows UIs to display thinking in real-time (e.g., collapsible panel showing "model is thinking...").

### Reference

**Best implementation:** pi-mono (`types.ts` lines 185-197)
```typescript
export type AssistantMessageEvent =
  | { type: "start"; partial: AssistantMessage }
  | { type: "text_start"; contentIndex: number; partial: AssistantMessage }
  | { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
  | { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
  | { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
  | { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
  | { type: "thinking_end"; contentIndex: number; content: string; partial: AssistantMessage }
  | { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
  | { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
  | { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
  | { type: "done"; reason: Extract<StopReason, "stop" | "length" | "toolUse">; message: AssistantMessage }
  | { type: "error"; reason: Extract<StopReason, "aborted" | "error">; error: AssistantMessage };
```

And in `types.ts` (lines 104-108):
```typescript
export interface ThinkingContent {
  type: "thinking";
  thinking: string;
  thinkingSignature?: string;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/types.py`
- `/home/dcramer/src/ash/src/ash/llm/anthropic.py`

### Proposed Changes

```python
# In types.py, add thinking types:

class StreamEventType(str, Enum):
    """Stream event type."""
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"
    # New thinking events
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"


@dataclass
class ThinkingContent:
    """Thinking content block from extended thinking."""
    thinking: str
    signature: str | None = None  # For caching/resumption
    type: ContentBlockType = ContentBlockType.TEXT  # Stored as text type


@dataclass
class StreamChunk:
    """A chunk from streaming response."""
    type: StreamEventType
    content: str | dict[str, Any] | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    content_index: int | None = None  # For correlating start/delta/end events
    thinking_signature: str | None = None  # For thinking blocks
```

```python
# In anthropic.py, modify stream() to emit thinking events (lines 237-309):

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
) -> AsyncGenerator[StreamChunk, None]:
    """Generate a streaming completion with thinking support."""
    model_name, kwargs = self._build_request_kwargs(
        messages, model, tools, system, max_tokens, temperature, thinking
    )

    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_content_index: int = 0
    in_thinking_block: bool = False

    assert self._semaphore is not None
    logger.debug(f"Waiting for API slot (stream, model={model_name})")
    async with self._semaphore:
        logger.debug(f"Acquired API slot, streaming {model_name}")
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    yield StreamChunk(type=StreamEventType.MESSAGE_START)

                elif event.type == "content_block_start":
                    current_content_index += 1

                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_START,
                            tool_use_id=current_tool_id,
                            tool_name=current_tool_name,
                            content_index=current_content_index,
                        )
                    elif event.content_block.type == "thinking":
                        # Extended thinking block started
                        in_thinking_block = True
                        yield StreamChunk(
                            type=StreamEventType.THINKING_START,
                            content_index=current_content_index,
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamChunk(
                            type=StreamEventType.TEXT_DELTA,
                            content=event.delta.text,
                            content_index=current_content_index,
                        )
                    elif event.delta.type == "thinking_delta":
                        # Extended thinking content
                        yield StreamChunk(
                            type=StreamEventType.THINKING_DELTA,
                            content=event.delta.thinking,
                            content_index=current_content_index,
                        )
                    elif event.delta.type == "input_json_delta":
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_DELTA,
                            content=event.delta.partial_json,
                            tool_use_id=current_tool_id,
                            content_index=current_content_index,
                        )

                elif event.type == "content_block_stop":
                    if current_tool_id:
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_END,
                            tool_use_id=current_tool_id,
                            content_index=current_content_index,
                        )
                        current_tool_id = None
                        current_tool_name = None
                    elif in_thinking_block:
                        yield StreamChunk(
                            type=StreamEventType.THINKING_END,
                            content_index=current_content_index,
                        )
                        in_thinking_block = False

                elif event.type == "message_stop":
                    yield StreamChunk(type=StreamEventType.MESSAGE_END)
        logger.debug("Stream complete")
```

### Effort

**S** (2-3 hours) - Adding new event types and handling them in stream processing.

### Priority

**Medium** - Enables richer UIs but not essential for core functionality. The thinking content is still available in non-streaming responses.

---

## Gap 4: Provider-Specific Rate Limit Handling with Auth Profile Rotation

### What Ash is Missing

Ash just retries with exponential backoff on rate limits. It has no concept of:
1. Multiple auth profiles per provider
2. Rotating to a different API key when one is rate-limited
3. Per-profile cooldown tracking
4. Usage-based round-robin selection

Clawdbot maintains a rich auth profile system:
- Multiple API keys per provider
- Round-robin selection based on `lastUsed`
- Cooldown tracking when profiles hit rate limits
- Automatic rotation on rate limit errors

### Reference

**Best implementation:** clawdbot (`auth-profiles.ts` lines 719-988)
```typescript
export type ProfileUsageStats = {
  lastUsed?: number;
  cooldownUntil?: number;
  disabledUntil?: number;
  disabledReason?: AuthProfileFailureReason;
  errorCount?: number;
  failureCounts?: Partial<Record<AuthProfileFailureReason, number>>;
  lastFailureAt?: number;
};

export function isProfileInCooldown(store: AuthProfileStore, profileId: string): boolean {
  const stats = store.usageStats?.[profileId];
  if (!stats) return false;
  const unusableUntil = resolveProfileUnusableUntil(stats);
  return unusableUntil ? Date.now() < unusableUntil : false;
}

export async function markAuthProfileUsed(params: {...}): Promise<void> {
  // Reset error count and update lastUsed
  freshStore.usageStats[profileId] = {
    ...freshStore.usageStats[profileId],
    lastUsed: Date.now(),
    errorCount: 0,
    cooldownUntil: undefined,
    disabledUntil: undefined,
  };
}

export function calculateAuthProfileCooldownMs(errorCount: number): number {
  const normalized = Math.max(1, errorCount);
  return Math.min(
    60 * 60 * 1000, // 1 hour max
    60 * 1000 * 5 ** Math.min(normalized - 1, 3), // 1min, 5min, 25min, 1hr
  );
}

export function resolveAuthProfileOrder(params: {...}): string[] {
  // Sort by type preference (oauth > token > api_key)
  // Then by lastUsed (oldest first for round-robin)
  // Put cooldown profiles at the end
  // ...
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/auth_profiles.py` (new file)
- `/home/dcramer/src/ash/src/ash/config/models.py`

### Proposed Changes

```python
# New file: src/ash/llm/auth_profiles.py
"""Auth profile management with cooldown tracking for LLM providers."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuthProfileType(str, Enum):
    """Type of authentication credential."""
    API_KEY = "api_key"
    TOKEN = "token"
    OAUTH = "oauth"


class FailureReason(str, Enum):
    """Reason for auth profile failure."""
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    BILLING = "billing"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ProfileUsageStats:
    """Usage statistics for an auth profile."""
    last_used: float | None = None
    cooldown_until: float | None = None
    disabled_until: float | None = None
    disabled_reason: FailureReason | None = None
    error_count: int = 0
    failure_counts: dict[FailureReason, int] = field(default_factory=dict)
    last_failure_at: float | None = None


@dataclass
class AuthProfile:
    """An authentication profile for an LLM provider."""
    profile_id: str
    provider: str
    profile_type: AuthProfileType
    key: str  # API key, token, or access token
    email: str | None = None
    expires: float | None = None  # For tokens with expiry


@dataclass
class AuthProfileStore:
    """Store for auth profiles with usage tracking."""
    profiles: dict[str, AuthProfile] = field(default_factory=dict)
    usage_stats: dict[str, ProfileUsageStats] = field(default_factory=dict)
    order: dict[str, list[str]] = field(default_factory=dict)  # provider -> profile order
    last_good: dict[str, str] = field(default_factory=dict)  # provider -> last good profile
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def add_profile(self, profile: AuthProfile) -> None:
        """Add or update an auth profile."""
        self.profiles[profile.profile_id] = profile
        if profile.profile_id not in self.usage_stats:
            self.usage_stats[profile.profile_id] = ProfileUsageStats()

    def get_profiles_for_provider(self, provider: str) -> list[AuthProfile]:
        """Get all profiles for a provider."""
        return [p for p in self.profiles.values() if p.provider == provider]

    def is_in_cooldown(self, profile_id: str) -> bool:
        """Check if a profile is currently in cooldown."""
        stats = self.usage_stats.get(profile_id)
        if not stats:
            return False

        now = time.time()

        if stats.cooldown_until and now < stats.cooldown_until:
            return True
        if stats.disabled_until and now < stats.disabled_until:
            return True

        return False

    def get_cooldown_remaining(self, profile_id: str) -> float:
        """Get seconds remaining in cooldown (0 if not in cooldown)."""
        stats = self.usage_stats.get(profile_id)
        if not stats:
            return 0.0

        now = time.time()
        remaining = 0.0

        if stats.cooldown_until:
            remaining = max(remaining, stats.cooldown_until - now)
        if stats.disabled_until:
            remaining = max(remaining, stats.disabled_until - now)

        return max(0.0, remaining)

    async def mark_used(self, profile_id: str) -> None:
        """Mark a profile as successfully used."""
        async with self._lock:
            if profile_id not in self.usage_stats:
                self.usage_stats[profile_id] = ProfileUsageStats()

            stats = self.usage_stats[profile_id]
            stats.last_used = time.time()
            stats.error_count = 0
            stats.cooldown_until = None
            stats.disabled_until = None
            stats.disabled_reason = None
            stats.failure_counts.clear()

    async def mark_failure(
        self,
        profile_id: str,
        reason: FailureReason,
        *,
        failure_window_hours: float = 24,
        billing_backoff_hours: float = 5,
        billing_max_hours: float = 24,
    ) -> None:
        """Mark a profile as failed and apply cooldown."""
        async with self._lock:
            if profile_id not in self.usage_stats:
                self.usage_stats[profile_id] = ProfileUsageStats()

            stats = self.usage_stats[profile_id]
            now = time.time()

            # Check if we should reset error count (failure window expired)
            failure_window_seconds = failure_window_hours * 3600
            if stats.last_failure_at and (now - stats.last_failure_at) > failure_window_seconds:
                stats.error_count = 0
                stats.failure_counts.clear()

            stats.error_count += 1
            stats.failure_counts[reason] = stats.failure_counts.get(reason, 0) + 1
            stats.last_failure_at = now

            if reason == FailureReason.BILLING:
                # Billing failures get longer backoff
                billing_count = stats.failure_counts.get(FailureReason.BILLING, 1)
                backoff_seconds = self._calculate_billing_backoff(
                    billing_count,
                    billing_backoff_hours * 3600,
                    billing_max_hours * 3600,
                )
                stats.disabled_until = now + backoff_seconds
                stats.disabled_reason = FailureReason.BILLING
            else:
                # Regular cooldown with exponential backoff
                backoff_seconds = self._calculate_cooldown(stats.error_count)
                stats.cooldown_until = now + backoff_seconds

    def _calculate_cooldown(self, error_count: int) -> float:
        """Calculate cooldown duration based on error count.

        Returns seconds: 60, 300, 1500, 3600 (1min, 5min, 25min, 1hr max)
        """
        normalized = max(1, error_count)
        return min(
            3600,  # 1 hour max
            60 * (5 ** min(normalized - 1, 3)),
        )

    def _calculate_billing_backoff(
        self,
        error_count: int,
        base_seconds: float,
        max_seconds: float,
    ) -> float:
        """Calculate billing failure backoff with exponential increase."""
        normalized = max(1, error_count)
        exponent = min(normalized - 1, 10)
        raw = base_seconds * (2 ** exponent)
        return min(max_seconds, raw)

    def resolve_profile_order(
        self,
        provider: str,
        *,
        preferred_profile: str | None = None,
    ) -> list[str]:
        """Get ordered list of profile IDs for a provider.

        Order: available profiles sorted by lastUsed (round-robin),
        then cooldown profiles sorted by cooldown expiry.
        """
        profiles = self.get_profiles_for_provider(provider)
        if not profiles:
            return []

        # Explicit order if configured
        explicit_order = self.order.get(provider)
        if explicit_order:
            profile_ids = [p for p in explicit_order if p in self.profiles]
        else:
            profile_ids = [p.profile_id for p in profiles]

        now = time.time()
        available: list[tuple[str, float]] = []
        in_cooldown: list[tuple[str, float]] = []

        for profile_id in profile_ids:
            stats = self.usage_stats.get(profile_id, ProfileUsageStats())

            cooldown_end = max(
                stats.cooldown_until or 0,
                stats.disabled_until or 0,
            )

            if cooldown_end > now:
                in_cooldown.append((profile_id, cooldown_end))
            else:
                # Sort by type preference then lastUsed
                profile = self.profiles.get(profile_id)
                type_score = {
                    AuthProfileType.OAUTH: 0,
                    AuthProfileType.TOKEN: 1,
                    AuthProfileType.API_KEY: 2,
                }.get(profile.profile_type if profile else AuthProfileType.API_KEY, 3)
                last_used = stats.last_used or 0
                # Combine type score (primary) and last_used (secondary)
                sort_key = type_score * 1e12 + last_used
                available.append((profile_id, sort_key))

        # Sort available by type then lastUsed (oldest first for round-robin)
        available.sort(key=lambda x: x[1])
        # Sort cooldown by expiry (soonest first)
        in_cooldown.sort(key=lambda x: x[1])

        result = [p[0] for p in available] + [p[0] for p in in_cooldown]

        # Move preferred profile to front if specified
        if preferred_profile and preferred_profile in result:
            result.remove(preferred_profile)
            result.insert(0, preferred_profile)

        return result

    async def get_api_key(
        self,
        provider: str,
        *,
        preferred_profile: str | None = None,
    ) -> tuple[str, str] | None:
        """Get the best available API key for a provider.

        Returns (api_key, profile_id) or None if no profiles available.
        """
        profile_order = self.resolve_profile_order(provider, preferred_profile=preferred_profile)

        for profile_id in profile_order:
            profile = self.profiles.get(profile_id)
            if not profile:
                continue

            # Skip expired tokens
            if profile.expires and profile.expires < time.time():
                continue

            # Return even if in cooldown (caller may want to wait or try anyway)
            return (profile.key, profile_id)

        return None

    def save(self, path: Path) -> None:
        """Save store to JSON file."""
        data = {
            "profiles": {
                pid: {
                    "profile_id": p.profile_id,
                    "provider": p.provider,
                    "profile_type": p.profile_type.value,
                    "key": p.key,
                    "email": p.email,
                    "expires": p.expires,
                }
                for pid, p in self.profiles.items()
            },
            "usage_stats": {
                pid: {
                    "last_used": s.last_used,
                    "cooldown_until": s.cooldown_until,
                    "disabled_until": s.disabled_until,
                    "disabled_reason": s.disabled_reason.value if s.disabled_reason else None,
                    "error_count": s.error_count,
                    "failure_counts": {k.value: v for k, v in s.failure_counts.items()},
                    "last_failure_at": s.last_failure_at,
                }
                for pid, s in self.usage_stats.items()
            },
            "order": self.order,
            "last_good": self.last_good,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "AuthProfileStore":
        """Load store from JSON file."""
        if not path.exists():
            return cls()

        data = json.loads(path.read_text())
        store = cls()

        for pid, pdata in data.get("profiles", {}).items():
            store.profiles[pid] = AuthProfile(
                profile_id=pdata["profile_id"],
                provider=pdata["provider"],
                profile_type=AuthProfileType(pdata["profile_type"]),
                key=pdata["key"],
                email=pdata.get("email"),
                expires=pdata.get("expires"),
            )

        for pid, sdata in data.get("usage_stats", {}).items():
            store.usage_stats[pid] = ProfileUsageStats(
                last_used=sdata.get("last_used"),
                cooldown_until=sdata.get("cooldown_until"),
                disabled_until=sdata.get("disabled_until"),
                disabled_reason=FailureReason(sdata["disabled_reason"]) if sdata.get("disabled_reason") else None,
                error_count=sdata.get("error_count", 0),
                failure_counts={FailureReason(k): v for k, v in sdata.get("failure_counts", {}).items()},
                last_failure_at=sdata.get("last_failure_at"),
            )

        store.order = data.get("order", {})
        store.last_good = data.get("last_good", {})

        return store
```

### Effort

**L** (1-2 days) - Complete new subsystem with persistence, locking, and integration.

### Priority

**Medium** - Very useful for production systems with multiple API keys, but single-key setups work fine without this.

---

## Gap 5: Model Context Window Tracking

### What Ash is Missing

Ash has no concept of context window size per model. It just sends requests and lets them fail if too large. This leads to:
1. Poor error messages when context is exceeded
2. No proactive truncation or warning
3. No way to select models based on context needs

Clawdbot/pi-mono track context window per model:

```typescript
export interface Model<TApi extends Api> {
  // ...
  contextWindow: number;  // Max tokens for context
  maxTokens: number;      // Max output tokens
}
```

### Reference

**Best implementation:** pi-mono (`types.ts` lines 225-244)
```typescript
export interface Model<TApi extends Api> {
  id: string;
  name: string;
  api: TApi;
  provider: Provider;
  baseUrl: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;  // Max input tokens
  maxTokens: number;      // Max output tokens
  headers?: Record<string, string>;
  compat?: TApi extends "openai-completions" ? OpenAICompat : never;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/models.py` (new file)
- `/home/dcramer/src/ash/src/ash/llm/base.py`

### Proposed Changes

```python
# New file: src/ash/llm/models.py
"""Model definitions with capabilities and limits."""

from dataclasses import dataclass


@dataclass
class ModelCost:
    """Cost per million tokens."""
    input: float  # $/M tokens
    output: float  # $/M tokens
    cache_read: float = 0.0  # $/M tokens for cached reads
    cache_write: float = 0.0  # $/M tokens for cache writes


@dataclass
class ModelInfo:
    """Model capabilities and limits."""
    id: str
    name: str
    provider: str
    context_window: int  # Max input tokens
    max_output_tokens: int  # Max output tokens
    supports_vision: bool = False
    supports_thinking: bool = False
    cost: ModelCost | None = None


# Known model definitions
MODELS: dict[str, ModelInfo] = {
    # Anthropic models
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude 4 Sonnet",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_thinking=True,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_thinking=True,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_thinking=False,
        cost=ModelCost(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
    ),
    # OpenAI models
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_thinking=False,
        cost=ModelCost(input=2.5, output=10.0),
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_thinking=False,
        cost=ModelCost(input=0.15, output=0.6),
    ),
    "o1": ModelInfo(
        id="o1",
        name="o1",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        supports_vision=True,
        supports_thinking=True,
        cost=ModelCost(input=15.0, output=60.0),
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        name="o1 Mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=65536,
        supports_vision=False,
        supports_thinking=True,
        cost=ModelCost(input=3.0, output=12.0),
    ),
}


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get model info by ID."""
    return MODELS.get(model_id)


def get_context_window(model_id: str, default: int = 128000) -> int:
    """Get context window size for a model."""
    info = get_model_info(model_id)
    return info.context_window if info else default


def get_max_output_tokens(model_id: str, default: int = 4096) -> int:
    """Get max output tokens for a model."""
    info = get_model_info(model_id)
    return info.max_output_tokens if info else default


def validate_context_size(
    model_id: str,
    input_tokens: int,
    requested_output_tokens: int,
) -> tuple[bool, str | None]:
    """Validate that a request fits within model limits.

    Returns (is_valid, error_message).
    """
    info = get_model_info(model_id)
    if not info:
        return True, None  # Unknown model, allow request

    total = input_tokens + requested_output_tokens

    if input_tokens > info.context_window:
        return False, (
            f"Input ({input_tokens:,} tokens) exceeds {info.name} context window "
            f"({info.context_window:,} tokens)"
        )

    if requested_output_tokens > info.max_output_tokens:
        return False, (
            f"Requested output ({requested_output_tokens:,} tokens) exceeds "
            f"{info.name} max output ({info.max_output_tokens:,} tokens)"
        )

    if total > info.context_window:
        available = info.context_window - input_tokens
        return False, (
            f"Request too large: {input_tokens:,} input + {requested_output_tokens:,} output "
            f"= {total:,} tokens exceeds {info.context_window:,} limit. "
            f"Available for output: {available:,} tokens"
        )

    return True, None
```

```python
# In base.py, add context validation helper:

from ash.llm.models import get_context_window, validate_context_size


class LLMProvider(ABC):
    # ... existing code ...

    def get_context_window(self, model: str | None = None) -> int:
        """Get context window size for a model."""
        model_id = model or self.default_model
        return get_context_window(model_id)

    def validate_request_size(
        self,
        model: str | None,
        input_tokens: int,
        max_tokens: int,
    ) -> tuple[bool, str | None]:
        """Validate request fits in context window."""
        model_id = model or self.default_model
        return validate_context_size(model_id, input_tokens, max_tokens)
```

### Effort

**S** (2-3 hours) - Static model data plus simple validation functions.

### Priority

**Low** - Nice for proactive error handling but API errors are usually clear enough. Becomes more important with automatic context management.

---

## Gap 6: Per-Provider/Model Cooldown Tracking

### What Ash is Missing

Ash's retry mechanism has no memory across requests. Each request starts fresh, unaware of recent failures. This means:
1. If Anthropic returns 429, the next request immediately hits Anthropic again
2. No backoff accumulation across requests
3. No provider-level cooldown (only per-request retry delay)

Clawdbot tracks cooldowns per profile (which maps to provider + API key):
- Error count persisted across requests
- Exponential backoff based on accumulated errors
- Cooldown expiry stored with timestamp
- Profiles automatically rotated away when in cooldown

### Reference

**Best implementation:** clawdbot (`auth-profiles.ts` lines 778-908)
```typescript
export function calculateAuthProfileCooldownMs(errorCount: number): number {
  const normalized = Math.max(1, errorCount);
  return Math.min(
    60 * 60 * 1000, // 1 hour max
    60 * 1000 * 5 ** Math.min(normalized - 1, 3),
  );
}

function computeNextProfileUsageStats(params: {...}): ProfileUsageStats {
  const windowMs = params.cfgResolved.failureWindowMs;
  const windowExpired =
    typeof params.existing.lastFailureAt === "number" &&
    params.existing.lastFailureAt > 0 &&
    params.now - params.existing.lastFailureAt > windowMs;

  const baseErrorCount = windowExpired ? 0 : (params.existing.errorCount ?? 0);
  const nextErrorCount = baseErrorCount + 1;
  // ... apply cooldown based on nextErrorCount
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/llm/cooldown.py` (new file)
- `/home/dcramer/src/ash/src/ash/llm/retry.py`

### Proposed Changes

```python
# New file: src/ash/llm/cooldown.py
"""Provider cooldown tracking for rate limiting."""

import asyncio
import time
from dataclasses import dataclass, field

from ash.llm.failover import FailoverReason


@dataclass
class CooldownStats:
    """Cooldown statistics for a provider/model."""
    error_count: int = 0
    last_failure_at: float | None = None
    cooldown_until: float | None = None
    failure_counts: dict[FailoverReason, int] = field(default_factory=dict)


class CooldownTracker:
    """Track cooldowns for providers and models.

    Maintains per-provider or per-model cooldown state to avoid
    hammering services that are rate limiting us.
    """

    def __init__(
        self,
        *,
        failure_window_hours: float = 24,
        max_cooldown_seconds: float = 3600,  # 1 hour
    ):
        """Initialize tracker.

        Args:
            failure_window_hours: Hours after which error count resets.
            max_cooldown_seconds: Maximum cooldown duration.
        """
        self._stats: dict[str, CooldownStats] = {}
        self._lock = asyncio.Lock()
        self._failure_window_seconds = failure_window_hours * 3600
        self._max_cooldown = max_cooldown_seconds

    def _get_key(self, provider: str, model: str | None = None) -> str:
        """Get cache key for provider/model."""
        if model:
            return f"{provider}:{model}"
        return provider

    def is_in_cooldown(self, provider: str, model: str | None = None) -> bool:
        """Check if a provider/model is in cooldown."""
        key = self._get_key(provider, model)
        stats = self._stats.get(key)
        if not stats or not stats.cooldown_until:
            return False
        return time.time() < stats.cooldown_until

    def get_cooldown_remaining(self, provider: str, model: str | None = None) -> float:
        """Get seconds remaining in cooldown (0 if not in cooldown)."""
        key = self._get_key(provider, model)
        stats = self._stats.get(key)
        if not stats or not stats.cooldown_until:
            return 0.0
        return max(0.0, stats.cooldown_until - time.time())

    async def record_success(self, provider: str, model: str | None = None) -> None:
        """Record a successful request, resetting cooldown."""
        key = self._get_key(provider, model)
        async with self._lock:
            self._stats[key] = CooldownStats()

    async def record_failure(
        self,
        provider: str,
        model: str | None = None,
        reason: FailoverReason | None = None,
    ) -> float:
        """Record a failed request and apply cooldown.

        Returns the cooldown duration in seconds.
        """
        key = self._get_key(provider, model)
        now = time.time()

        async with self._lock:
            stats = self._stats.get(key) or CooldownStats()

            # Reset if failure window expired
            if stats.last_failure_at:
                if (now - stats.last_failure_at) > self._failure_window_seconds:
                    stats = CooldownStats()

            stats.error_count += 1
            stats.last_failure_at = now

            if reason:
                stats.failure_counts[reason] = stats.failure_counts.get(reason, 0) + 1

            # Calculate cooldown: 60s, 300s, 1500s, 3600s (1min, 5min, 25min, 1hr)
            cooldown = min(
                self._max_cooldown,
                60 * (5 ** min(stats.error_count - 1, 3)),
            )
            stats.cooldown_until = now + cooldown

            self._stats[key] = stats
            return cooldown

    async def wait_if_needed(self, provider: str, model: str | None = None) -> None:
        """Wait for cooldown to expire if in cooldown."""
        remaining = self.get_cooldown_remaining(provider, model)
        if remaining > 0:
            await asyncio.sleep(remaining)


# Global tracker instance
_tracker: CooldownTracker | None = None


def get_cooldown_tracker() -> CooldownTracker:
    """Get the global cooldown tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CooldownTracker()
    return _tracker
```

```python
# In retry.py, integrate cooldown tracking:

from ash.llm.cooldown import get_cooldown_tracker
from ash.llm.failover import classify_failover_reason


async def with_retry[T](
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    operation_name: str = "API call",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute.
        config: Retry configuration.
        operation_name: Name for logging.
        provider: Provider name for cooldown tracking.
        model: Model name for cooldown tracking.
    """
    config = config or RetryConfig()
    tracker = get_cooldown_tracker()

    if not config.enabled:
        return await func()

    # Check if provider is in cooldown
    if provider and tracker.is_in_cooldown(provider, model):
        remaining = tracker.get_cooldown_remaining(provider, model)
        logger.warning(
            f"{operation_name}: {provider} is in cooldown for {remaining:.1f}s more"
        )
        # Optionally wait for cooldown
        # await tracker.wait_if_needed(provider, model)

    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            result = await func()
            # Success - reset cooldown
            if provider:
                await tracker.record_success(provider, model)
            return result
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                raise

            if attempt >= config.max_retries:
                logger.warning(
                    f"{operation_name} failed after {config.max_retries + 1} attempts: {e}"
                )
                # Record failure for cooldown tracking
                if provider:
                    reason = classify_failover_reason(e)
                    await tracker.record_failure(provider, model, reason)
                raise

            # Calculate delay
            delay_ms = min(
                config.base_delay_ms * (2**attempt),
                config.max_delay_ms,
            )
            delay_s = delay_ms / 1000

            logger.info(
                f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay_s:.1f}s: {e}"
            )

            await asyncio.sleep(delay_s)

    assert last_error is not None
    raise last_error
```

### Effort

**M** (half day) - New tracking module plus integration with retry logic.

### Priority

**Medium** - Improves behavior under sustained rate limiting, but simple retries often suffice for light usage.

---

## Summary Table

| Gap | Description | Effort | Priority | Main Benefit |
|-----|-------------|--------|----------|--------------|
| 1 | Model failover cascade | M | **High** | Automatic fallback when providers fail |
| 2 | Unified thinking level abstraction | S | Medium | Consistent reasoning across providers |
| 3 | Thinking delta events | S | Medium | Real-time thinking display in UIs |
| 4 | Auth profile rotation | L | Medium | Multiple API keys with automatic rotation |
| 5 | Context window tracking | S | Low | Proactive validation of request size |
| 6 | Per-provider cooldown tracking | M | Medium | Better rate limit handling across requests |

## Recommended Implementation Order

1. **Gap 1: Model failover cascade** (High priority, critical for reliability)
2. **Gap 6: Per-provider cooldown tracking** (Medium, improves rate limit handling)
3. **Gap 2: Unified thinking level abstraction** (Medium, enables OpenAI reasoning)
4. **Gap 3: Thinking delta events** (Medium, better UX for thinking models)
5. **Gap 4: Auth profile rotation** (Medium, useful for teams/production)
6. **Gap 5: Context window tracking** (Low, nice to have validation)
