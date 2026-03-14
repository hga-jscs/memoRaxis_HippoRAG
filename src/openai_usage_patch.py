from __future__ import annotations

import time
from typing import Any

from .token_tracker import get_global_tracker

_PATCHED = False


def _extract_prompt_chars_from_messages(messages: Any) -> int:
    if not isinstance(messages, list):
        return len(str(messages))
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            total += len(str(msg.get("content", "")))
        else:
            total += len(str(msg))
    return total


def install_openai_usage_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings

    real_chat_create = Completions.create
    real_embed_create = Embeddings.create

    def patched_chat_create(self, *args, **kwargs):
        tracker = get_global_tracker()
        t0 = time.perf_counter()
        resp = None
        ok = True
        try:
            resp = real_chat_create(self, *args, **kwargs)
            return resp
        except Exception:
            ok = False
            raise
        finally:
            if tracker and ok and resp is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(
                    getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
                )

                messages = kwargs.get("messages", [])
                prompt_chars = _extract_prompt_chars_from_messages(messages)
                content = ""
                try:
                    content = resp.choices[0].message.content or ""
                except Exception:
                    content = ""

                tracker.record(
                    provider="openai_compat",
                    api_kind="chat",
                    model=kwargs.get("model", ""),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    prompt_chars=prompt_chars,
                    output_chars=len(content),
                    latency_ms=latency_ms,
                    ok=True,
                )

    def patched_embed_create(self, *args, **kwargs):
        tracker = get_global_tracker()
        t0 = time.perf_counter()
        resp = None
        ok = True
        try:
            resp = real_embed_create(self, *args, **kwargs)
            return resp
        except Exception:
            ok = False
            raise
        finally:
            if tracker and ok and resp is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", prompt_tokens) or 0)

                inp = kwargs.get("input", "")
                if isinstance(inp, list):
                    prompt_chars = sum(len(str(x)) for x in inp)
                else:
                    prompt_chars = len(str(inp))

                tracker.record(
                    provider="openai_compat",
                    api_kind="embedding",
                    model=kwargs.get("model", ""),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens,
                    prompt_chars=prompt_chars,
                    output_chars=0,
                    latency_ms=latency_ms,
                    ok=True,
                )

    Completions.create = patched_chat_create
    Embeddings.create = patched_embed_create
    _PATCHED = True
