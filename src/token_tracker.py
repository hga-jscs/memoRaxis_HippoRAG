from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


_scope_var = contextvars.ContextVar("token_scope", default={})


@dataclass
class TokenEvent:
    run_id: str
    ts: float
    dataset: str = ""
    instance_idx: Optional[int] = None
    question_idx: Optional[int] = None
    adaptor: str = ""
    stage: str = ""
    substage: str = ""
    api_kind: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class TokenTracker:
    def __init__(self, out_dir: str = "out/token_traces", run_id: Optional[str] = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.run_id}.jsonl"

    @contextmanager
    def scope(self, **kwargs: Any) -> Iterator[None]:
        old = _scope_var.get()
        new = dict(old)
        new.update({k: v for k, v in kwargs.items() if v is not None})
        token = _scope_var.set(new)
        try:
            yield
        finally:
            _scope_var.reset(token)

    def record(self, **kwargs: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "ts": time.time(),
            **_scope_var.get(),
            **kwargs,
        }
        event = TokenEvent(**payload)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


_GLOBAL_TRACKER: Optional[TokenTracker] = None


def set_global_tracker(tracker: Optional[TokenTracker]) -> None:
    global _GLOBAL_TRACKER
    _GLOBAL_TRACKER = tracker


def get_global_tracker() -> Optional[TokenTracker]:
    return _GLOBAL_TRACKER
