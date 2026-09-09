"""Provider-agnostic helpers for parsing streamed LLM message content.

LangChain's `AIMessageChunk.content` shape differs by provider:
  - OpenAI:    str — `"Hello"`
  - Anthropic: list[dict] — `[{"type": "text", "text": "Hello"}, ...]`
                            also `tool_use` / `thinking` block types we skip.

`iter_text(chunk)` normalizes both to a flat string-yielding iterator so
the agent's stream loop doesn't have to branch on provider.
"""

from typing import Iterable

from langchain_core.messages import AIMessageChunk

TRUNCATION_NOTICE = "\n\n[response truncated - output token limit reached]"


def response_stop_reason(message) -> str | None:
    """Return a message's provider stop reason."""
    if isinstance(message, dict):
        metadata = message.get("response_metadata")
    else:
        metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    return metadata.get("stop_reason") or metadata.get("finish_reason")


def finalize_response_text(text: str, stop_reason: str | None) -> str:
    """Make a token-limited response explicit and syntactically closed."""
    if stop_reason != "max_tokens":
        return text
    suffix = TRUNCATION_NOTICE
    if text.count("```") % 2:
        suffix += "\n```"
    return text + suffix


def iter_text(chunk: AIMessageChunk) -> Iterable[str]:
    """Yield the user-visible text fragments from one AIMessageChunk.

    Skips tool-use / thinking / image blocks. Returns nothing for chunks
    that don't carry text (e.g. tool-call-only chunks).
    """
    content = chunk.content
    if isinstance(content, str):
        if content:
            yield content
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text") or ""
                if text:
                    yield text
